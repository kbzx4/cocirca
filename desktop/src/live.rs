//! Live recording from a phone running Phyphox ("Allow remote access"), or from a simulated phone.
//!
//! Phyphox serves `GET /config` (experiment title and buffer names) and `GET /get?buf=threshold|timebuf`
//! (the samples of `buf` whose time-buffer value is above `threshold`). Each poll fetches the samples
//! recorded since the previous poll and reduces them to one row: activity = mean ENMO in milli-g over
//! those samples, light = mean illuminance. Rows are appended to a CSV log as they arrive, so a
//! multi-day recording survives closing the app and can be resumed.

use crate::data::{ms_to_naive, naive_to_ms, Channel, RawDataset, Role};
use eframe::egui;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const G: f64 = 9.80665;
pub const LOG_HEADER: &str = "timestamp,Activity (mg),Illuminance (lx)";
/// Log header when the light column is camera luminance (Phyphox "Brightness" experiment)
pub const LOG_HEADER_LUMINANCE: &str = "timestamp,Activity (mg),Luminance (cd/m2)";
const TS_FMT: &str = "%Y-%m-%dT%H:%M:%S";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub t_ms: i64,
    /// mean ENMO over the samples since the previous poll, milli-g (NaN: no accelerometer data)
    pub activity: f64,
    /// mean illuminance, lux (NaN: no light data)
    pub lux: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Phone,
    /// Simulated phone; `speed` simulated seconds pass per real second (1 = real time)
    Simulated { speed: u32 },
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub url: String,
    pub source: Source,
    pub poll_s: f64,
    pub log_path: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { url: "http://192.168.1.50:8080".into(), source: Source::Phone, poll_s: 5.0, log_path: default_log_path() }
    }
}

/// `Documents\CoSirca\phyphox_live_YYYY-MM-DD_HHMM.csv`
pub fn default_log_path() -> PathBuf {
    let base = std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join("Documents")).unwrap_or_else(|| PathBuf::from("."));
    base.join("CoSirca").join(format!("phyphox_live_{}.csv", chrono::Local::now().format("%Y-%m-%d_%H%M")))
}

// ------------------------------------------------------------------------------------------------
// Phyphox buffers
// ------------------------------------------------------------------------------------------------

/// Which Phyphox buffers hold acceleration x / y / z, light, and their time stamps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Buffers {
    pub ax: Option<String>,
    pub ay: Option<String>,
    pub az: Option<String>,
    pub acc_time: Option<String>,
    pub lux: Option<String>,
    pub lux_time: Option<String>,
    /// light is camera luminance (cd/m², "Brightness" experiment) rather than illuminance (lux)
    pub luminance: bool,
}

impl Buffers {
    pub fn from_names(names: &[String]) -> Buffers {
        let mut b = Buffers::default();
        let mut spare_time: Vec<&String> = Vec::new();
        // camera "Brightness (Luminance)" experiment: luminance (cd/m²), else relative luma
        let (mut lumin, mut luma): (Option<String>, Option<String>) = (None, None);
        for n in names {
            let low = n.to_lowercase();
            let is_time = low.contains("time") || low == "t" || low.ends_with("_t");
            let light = low.contains("illum") || low.contains("lux") || low.contains("light");
            if !is_time && !light {
                if low.starts_with("luminance") {
                    lumin.get_or_insert_with(|| n.clone());
                    continue;
                } else if low == "luma" {
                    luma.get_or_insert_with(|| n.clone());
                    continue;
                }
            }
            let acc = low.starts_with("acc") || low.starts_with("lin") || low.contains("accel") || matches!(low.as_str(), "ax" | "ay" | "az");
            if is_time {
                if acc && b.acc_time.is_none() {
                    b.acc_time = Some(n.clone());
                } else if light && b.lux_time.is_none() {
                    b.lux_time = Some(n.clone());
                } else {
                    spare_time.push(n);
                }
            } else if light {
                b.lux.get_or_insert_with(|| n.clone());
            } else if acc {
                let slot = match low.chars().last() {
                    Some('x') => &mut b.ax,
                    Some('y') => &mut b.ay,
                    Some('z') => &mut b.az,
                    _ => continue,
                };
                slot.get_or_insert_with(|| n.clone());
            }
        }
        if b.lux.is_none() {
            if let Some(l) = lumin.or(luma) {
                b.lux = Some(l);
                b.luminance = true;
            }
        }
        // a generic time buffer ("t", "time", or the only spare one) serves whichever group has none
        let generic = spare_time.iter().find(|t| matches!(t.to_lowercase().as_str(), "t" | "time")).or(if spare_time.len() == 1 { spare_time.first() } else { None });
        if let Some(t) = generic {
            if b.has_accel() && b.acc_time.is_none() {
                b.acc_time = Some((*t).clone());
            }
            if b.lux.is_some() && b.lux_time.is_none() {
                b.lux_time = Some((*t).clone());
            }
        }
        b
    }

    pub fn has_accel(&self) -> bool {
        self.ax.is_some() && self.ay.is_some() && self.az.is_some()
    }

    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.has_accel() {
            parts.push(format!(
                "acceleration: {}, {}, {}{}",
                self.ax.as_deref().unwrap_or(""),
                self.ay.as_deref().unwrap_or(""),
                self.az.as_deref().unwrap_or(""),
                self.acc_time.as_ref().map(|t| format!(" (time: {t})")).unwrap_or_else(|| " (no time buffer: latest value only)".into())
            ));
        }
        if let Some(l) = &self.lux {
            let kind = if self.luminance { "camera brightness" } else { "light" };
            parts.push(format!("{kind}: {l}{}", self.lux_time.as_ref().map(|t| format!(" (time: {t})")).unwrap_or_else(|| " (latest value only)".into())));
        }
        parts.join(" · ")
    }
}

#[derive(Clone, Debug)]
pub struct Probe {
    pub title: String,
    pub buffers: Vec<String>,
    pub map: Buffers,
}

/// Read the experiment's `/config` and find the acceleration / light buffers.
pub fn probe(url: &str) -> Result<Probe, String> {
    let body = http_get(url, "/config", Duration::from_secs(4))?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|_| "The answer from /config is not Phyphox JSON. Is this the address Phyphox shows under \"Allow remote access\"?".to_string())?;
    let title = v["title"].as_str().unwrap_or("Phyphox experiment").to_string();
    let buffers: Vec<String> = v["buffers"].as_array().map(|a| a.iter().filter_map(|b| b["name"].as_str().map(String::from)).collect()).unwrap_or_default();
    let map = Buffers::from_names(&buffers);
    if map.lux.is_none() && !map.has_accel() {
        return Err(format!(
            "The experiment \"{title}\" has no light or 3-axis acceleration buffers (found: {}). Open an experiment with Acceleration and/or Light in Phyphox.",
            if buffers.is_empty() { "none".to_string() } else { buffers.join(", ") }
        ));
    }
    Ok(Probe { title, buffers, map })
}

// ------------------------------------------------------------------------------------------------
// Minimal HTTP/1.1 GET (Phyphox serves plain http on the local network)
// ------------------------------------------------------------------------------------------------

fn split_url(url: &str) -> Result<(String, String), String> {
    let u = url.trim();
    if u.to_lowercase().starts_with("https://") {
        return Err("Phyphox serves http://, not https://".into());
    }
    let u = u.strip_prefix("http://").or_else(|| u.strip_prefix("HTTP://")).unwrap_or(u);
    let (hp, path) = match u.find('/') {
        Some(i) => (&u[..i], u[i..].trim_end_matches('/')),
        None => (u, ""),
    };
    if hp.is_empty() {
        return Err("Enter the address Phyphox shows, e.g. http://192.168.1.50:8080".into());
    }
    let hp = if hp.contains(':') { hp.to_string() } else { format!("{hp}:80") };
    Ok((hp, path.to_string()))
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

/// Whether a full response (headers + body) has arrived, from Content-Length or the chunked terminator.
fn complete(buf: &[u8]) -> bool {
    let Some(h) = find(buf, b"\r\n\r\n") else { return false };
    let head = String::from_utf8_lossy(&buf[..h]).to_lowercase();
    if let Some(cl) = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()) {
        return buf.len() >= h + 4 + cl;
    }
    head.contains("transfer-encoding: chunked") && buf.ends_with(b"0\r\n\r\n")
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(i) = find(b, b"\r\n") {
        let size = usize::from_str_radix(String::from_utf8_lossy(&b[..i]).split(';').next().unwrap_or("0").trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = i + 2;
        let end = (start + size).min(b.len());
        out.extend_from_slice(&b[start..end]);
        b = &b[(end + 2).min(b.len())..];
    }
    out
}

pub fn http_get(url: &str, path_query: &str, timeout: Duration) -> Result<String, String> {
    let (hp, prefix) = split_url(url)?;
    let addr = hp.to_socket_addrs().map_err(|e| format!("{hp}: {e}"))?.next().ok_or_else(|| format!("{hp}: address not found"))?;
    let mut s = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("Cannot reach {hp} ({e}). Check the address, that the phone is on the same Wi-Fi, and that Phyphox shows \"Remote access\" as on."))?;
    s.set_read_timeout(Some(timeout)).ok();
    s.set_write_timeout(Some(timeout)).ok();
    let host = hp.strip_suffix(":80").unwrap_or(&hp);
    // one write: some small embedded servers read the request in a single call
    let req = format!("GET {prefix}{path_query} HTTP/1.1\r\nHost: {host}\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).map_err(|e| format!("{hp}: {e}"))?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 64 * 1024];
    loop {
        match s.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if complete(&buf) {
                    break;
                }
            }
            Err(_) if complete(&buf) => break,
            Err(e) => return Err(format!("{hp}: no answer ({e})")),
        }
    }
    let h = find(&buf, b"\r\n\r\n").ok_or_else(|| format!("{hp}: incomplete answer"))?;
    let head = String::from_utf8_lossy(&buf[..h]).to_string();
    let status = head.lines().next().unwrap_or("");
    if !status.split_whitespace().nth(1).is_some_and(|c| c == "200") {
        return Err(format!("{hp}{prefix}{}: {status}", path_query.split('?').next().unwrap_or("")));
    }
    let body = &buf[h + 4..];
    let body = if head.to_lowercase().contains("transfer-encoding: chunked") { dechunk(body) } else { body.to_vec() };
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Start (`"start"`), stop or clear the measurement on the phone (Phyphox `/control?cmd=…`).
pub fn control(url: &str, cmd: &str) -> Result<(), String> {
    let body = http_get(url, &format!("/control?cmd={}", enc(cmd)), Duration::from_secs(4))?;
    let ok = serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|v| v["result"].as_bool()).unwrap_or(false);
    if ok {
        Ok(())
    } else {
        Err(format!("Phyphox did not accept \"{cmd}\""))
    }
}

fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Polling
// ------------------------------------------------------------------------------------------------

#[derive(Default)]
struct PhoneState {
    map: Buffers,
    last_acc_t: Option<f64>,
    last_lux_t: Option<f64>,
    session: Option<String>,
    /// gravity in the acceleration buffers: 9.81 (with g) or 0 (linear acceleration), judged once
    g: Option<f64>,
}

struct Poll {
    activity: f64,
    lux: f64,
    measuring: Option<bool>,
}

fn nums(v: &serde_json::Value, name: &Option<String>) -> Vec<f64> {
    let Some(n) = name else { return Vec::new() };
    v["buffer"][n.as_str()]["buffer"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default()
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = v.filter(|x| x.is_finite()).fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n > 0 {
        s / n as f64
    } else {
        f64::NAN
    }
}

fn poll_phone(url: &str, st: &mut PhoneState) -> Result<Poll, String> {
    let mut q: Vec<String> = Vec::new();
    // samples newer than the last poll when the group has a time buffer, else the latest value
    let ask = |q: &mut Vec<String>, name: &Option<String>, time: &Option<String>, last: Option<f64>| {
        if let Some(n) = name {
            q.push(match (time, last) {
                (Some(tb), Some(t)) => format!("{}={}", enc(n), enc(&format!("{t}|{tb}"))),
                _ => enc(n),
            });
        }
    };
    if st.map.has_accel() {
        for b in [&st.map.ax, &st.map.ay, &st.map.az, &st.map.acc_time] {
            ask(&mut q, b, &st.map.acc_time, st.last_acc_t);
        }
    }
    for b in [&st.map.lux, &st.map.lux_time] {
        ask(&mut q, b, &st.map.lux_time, st.last_lux_t);
    }
    q.dedup();
    let body = http_get(url, &format!("/get?{}", q.join("&")), Duration::from_secs(6))?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|_| "Phyphox sent an unreadable answer to /get".to_string())?;
    let measuring = v["status"]["measuring"].as_bool();
    let session = v["status"]["session"].as_str().map(String::from);
    if session.is_some() && st.session.is_some() && session != st.session {
        // the experiment was restarted: its clock starts again from 0
        st.last_acc_t = None;
        st.last_lux_t = None;
    }
    st.session = session;
    let mut activity = f64::NAN;
    if st.map.has_accel() {
        let (x, y, z) = (nums(&v, &st.map.ax), nums(&v, &st.map.ay), nums(&v, &st.map.az));
        let n = x.len().min(y.len()).min(z.len());
        let vm: Vec<f64> = (0..n).map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt()).filter(|m| m.is_finite()).collect();
        if !vm.is_empty() {
            let g = *st.g.get_or_insert_with(|| if crate::stats::median(&vm) > 5.0 { G } else { 0.0 });
            activity = mean(vm.iter().map(|&m| (m - g).max(0.0) / G * 1000.0));
        }
        let t = nums(&v, &st.map.acc_time);
        if let Some(&tl) = t.iter().rev().find(|t| t.is_finite()) {
            if st.last_acc_t.is_some_and(|l| tl < l) {
                st.last_acc_t = None; // clock went backwards (cleared data)
            } else {
                st.last_acc_t = Some(tl);
            }
        }
    }
    let lux = mean(nums(&v, &st.map.lux).into_iter());
    if let Some(&tl) = nums(&v, &st.map.lux_time).iter().rev().find(|t| t.is_finite()) {
        st.last_lux_t = if st.last_lux_t.is_some_and(|l| tl < l) { None } else { Some(tl) };
    }
    Ok(Poll { activity, lux, measuring })
}

/// Simulated phone: a 24 h activity rhythm peaking mid-afternoon with daytime light, plus noise.
struct Sim {
    rng: u64,
}

impl Sim {
    fn normal(&mut self) -> f64 {
        let mut u = || {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            ((self.rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }

    fn row(&mut self, t_ms: i64) -> Poll {
        use chrono::Timelike;
        let t = ms_to_naive(t_ms);
        let tod = t.hour() as f64 + t.minute() as f64 / 60.0 + t.second() as f64 / 3600.0;
        let circ = 0.5 + 0.5 * (std::f64::consts::TAU * (tod - 14.5) / 24.0).cos();
        let awake = (std::f64::consts::TAU * (tod - 14.5) / 24.0).cos() > -0.45;
        let activity = (if awake { 8.0 + 55.0 * circ } else { 1.5 } + self.normal() * if awake { 12.0 } else { 0.8 }).max(0.0);
        let day = (std::f64::consts::TAU * (tod - 13.0) / 24.0).cos();
        let lux = if day > 0.0 { (600.0 * day + 40.0 * self.normal()).max(5.0) } else { (if awake { 80.0 } else { 0.5 } + self.normal().abs() * 2.0).max(0.0) };
        Poll { activity, lux, measuring: Some(true) }
    }
}

// ------------------------------------------------------------------------------------------------
// Log file
// ------------------------------------------------------------------------------------------------

fn fmt_num(v: f64, dp: usize) -> String {
    if v.is_finite() {
        format!("{v:.dp$}")
    } else {
        String::new()
    }
}

/// Read a log written by this recorder (to resume it). Returns the rows and whether the light column
/// is camera luminance.
pub fn read_log(path: &Path) -> Result<(Vec<Row>, bool), String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = BufReader::new(f).lines();
    let head = lines.next().transpose().map_err(|e| e.to_string())?.unwrap_or_default();
    let luminance = match head.trim_start_matches('\u{feff}').trim() {
        LOG_HEADER => false,
        LOG_HEADER_LUMINANCE => true,
        _ => return Err(format!("{} is not a CoSirca live log. Choose a new file to record into.", path.display())),
    };
    let num = |s: Option<&str>| s.map(str::trim).filter(|s| !s.is_empty()).and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN);
    let mut rows = Vec::new();
    for l in lines {
        let l = l.map_err(|e| e.to_string())?;
        let mut it = l.split(',');
        let Some(t) = it.next().and_then(|t| chrono::NaiveDateTime::parse_from_str(t.trim(), TS_FMT).ok()) else { continue };
        rows.push(Row { t_ms: naive_to_ms(t), activity: num(it.next()), lux: num(it.next()) });
    }
    Ok((rows, luminance))
}

fn open_log(path: &Path, luminance: bool) -> Result<File, String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let new = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if new {
        writeln!(f, "{}", if luminance { LOG_HEADER_LUMINANCE } else { LOG_HEADER }).map_err(|e| e.to_string())?;
    }
    Ok(f)
}

// ------------------------------------------------------------------------------------------------
// Recorder
// ------------------------------------------------------------------------------------------------

#[derive(Default)]
pub struct Status {
    pub rows: Vec<Row>,
    /// rows loaded from an existing log when recording resumed
    pub resumed: usize,
    pub polls: usize,
    pub last: Option<Row>,
    pub last_error: Option<String>,
    pub measuring: Option<bool>,
}

pub struct Recorder {
    pub settings: Settings,
    pub title: String,
    pub buffers: String,
    /// the light column is camera luminance (cd/m²) rather than illuminance (lux)
    pub luminance: bool,
    pub started: Instant,
    status: Arc<Mutex<Status>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Recorder {
    /// Connects (for a phone), resumes the log if it exists, and starts polling in the background.
    pub fn start(settings: Settings, ctx: egui::Context) -> Result<Recorder, String> {
        if !(settings.poll_s > 0.0) {
            return Err("The poll interval must be positive.".into());
        }
        let probe = match settings.source {
            Source::Phone => Some(probe(&settings.url)?),
            Source::Simulated { .. } => None,
        };
        let luminance = probe.as_ref().is_some_and(|p| p.map.luminance);
        let rows = if std::fs::metadata(&settings.log_path).is_ok_and(|m| m.len() > 0) {
            let (rows, file_lum) = read_log(&settings.log_path)?;
            if file_lum != luminance && rows.iter().any(|r| r.lux.is_finite()) {
                return Err(format!(
                    "{} holds {}, but this experiment gives {}. Choose a new file (New file).",
                    settings.log_path.display(),
                    if file_lum { "camera luminance (cd/m²)" } else { "illuminance (lux)" },
                    if luminance { "camera luminance (cd/m²)" } else { "illuminance (lux)" }
                ));
            }
            rows
        } else {
            Vec::new()
        };
        let mut file = open_log(&settings.log_path, luminance)?;
        let status = Arc::new(Mutex::new(Status { resumed: rows.len(), last: rows.last().copied(), rows, ..Default::default() }));
        let stop = Arc::new(AtomicBool::new(false));
        let (title, buffers) = match &probe {
            Some(p) => (p.title.clone(), p.map.describe()),
            None => ("Simulated phone".into(), "activity and light".into()),
        };
        let th = {
            let (settings, status, stop) = (settings.clone(), status.clone(), stop.clone());
            let mut phone = PhoneState { map: probe.map(|p| p.map).unwrap_or_default(), ..Default::default() };
            let mut sim = Sim { rng: 0x9E37_79B9_7F4A_7C15 ^ chrono::Local::now().timestamp_millis() as u64 };
            let now_ms = || naive_to_ms(chrono::Local::now().naive_local());
            // simulated clock: continues after the last logged row, or starts now
            let mut sim_t = status.lock().unwrap().last.map_or_else(now_ms, |r| r.t_ms);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    let (t_ms, res) = match settings.source {
                        Source::Phone => (now_ms(), poll_phone(&settings.url, &mut phone)),
                        Source::Simulated { speed: 1 } => {
                            let t = now_ms();
                            (t, Ok(sim.row(t)))
                        }
                        Source::Simulated { speed } => {
                            sim_t += (settings.poll_s * speed as f64 * 1000.0) as i64;
                            (sim_t, Ok(sim.row(sim_t)))
                        }
                    };
                    let mut st = status.lock().unwrap();
                    st.polls += 1;
                    match res {
                        Ok(p) => {
                            st.measuring = p.measuring;
                            st.last_error = None;
                            if p.activity.is_finite() || p.lux.is_finite() {
                                let row = Row { t_ms, activity: p.activity, lux: p.lux };
                                let line = format!("{},{},{}", ms_to_naive(t_ms).format(TS_FMT), fmt_num(p.activity, 3), fmt_num(p.lux, 2));
                                if let Err(e) = writeln!(file, "{line}").and_then(|_| file.flush()) {
                                    st.last_error = Some(format!("Cannot write the log: {e}"));
                                }
                                st.rows.push(row);
                                st.last = Some(row);
                            }
                        }
                        Err(e) => st.last_error = Some(e),
                    }
                    drop(st);
                    ctx.request_repaint();
                    let wait = Duration::from_secs_f64(settings.poll_s);
                    while t0.elapsed() < wait && !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(20).min(wait));
                    }
                }
            })
        };
        Ok(Recorder { settings, title, buffers, luminance, started: Instant::now(), status, stop, thread: Some(th) })
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Run `f` on the shared status (rows, counters, last error).
    pub fn with_status<T>(&self, f: impl FnOnce(&Status) -> T) -> T {
        f(&self.status.lock().unwrap())
    }

    /// The recording so far as a dataset for the analysis pipeline.
    pub fn dataset(&self) -> RawDataset {
        let st = self.status.lock().unwrap();
        let mut channels = Vec::new();
        let act: Vec<f64> = st.rows.iter().map(|r| r.activity).collect();
        if act.iter().any(|v| v.is_finite()) {
            channels.push(Channel::new(Role::Activity, "Activity", "mg", act));
        }
        let lux: Vec<f64> = st.rows.iter().map(|r| r.lux).collect();
        if lux.iter().any(|v| v.is_finite()) {
            channels.push(if self.luminance { Channel::new(Role::Light, "Luminance", "cd/m²", lux) } else { Channel::new(Role::Light, "Light", "lux", lux) });
        }
        let file = self.settings.log_path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let mut d = RawDataset {
            name: format!("Live · {file}"),
            format: match self.settings.source {
                Source::Phone => format!("Phyphox live recording ({})", self.title),
                Source::Simulated { .. } => "Simulated phone (live)".into(),
            },
            t_ms: st.rows.iter().map(|r| r.t_ms).collect(),
            channels,
            ..Default::default()
        };
        d.sort_by_time();
        d
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn buffer_mapping() {
        let b = Buffers::from_names(&names(&["accX", "accY", "accZ", "acc", "acc_time", "illum", "illum_time"]));
        assert_eq!((b.ax.as_deref(), b.az.as_deref(), b.acc_time.as_deref()), (Some("accX"), Some("accZ"), Some("acc_time")));
        assert_eq!((b.lux.as_deref(), b.lux_time.as_deref()), (Some("illum"), Some("illum_time")));
        let b = Buffers::from_names(&names(&["illum_time", "illum"]));
        assert_eq!((b.lux.as_deref(), b.lux_time.as_deref(), b.has_accel()), (Some("illum"), Some("illum_time"), false));
        let b = Buffers::from_names(&names(&["t", "linX", "linY", "linZ"]));
        assert_eq!(b.acc_time.as_deref(), Some("t"));
        // camera "Brightness (Luminance)" experiment, as served by a real phone
        let b = Buffers::from_names(&names(&["luma", "luminance", "t", "shutterSpeed", "iso", "aperture", "exposure", "lastTimes", "lastPeriods", "framePeriod", "frameRate"]));
        assert_eq!((b.lux.as_deref(), b.lux_time.as_deref(), b.luminance, b.has_accel()), (Some("luminance"), Some("t"), true, false));
    }

    /// A fake Phyphox: answers /config and /get, records the query strings it was sent.
    fn fake_phyphox(answers: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for body in answers {
                let (mut s, _) = l.accept().unwrap();
                let mut got = Vec::new();
                let mut buf = [0u8; 4096];
                while find(&got, b"\r\n\r\n").is_none() {
                    let n = s.read(&mut buf).unwrap();
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                }
                let req = String::from_utf8_lossy(&got).to_string();
                seen.push(req.lines().next().unwrap_or("").to_string());
                // chunked on purpose, to exercise the decoder
                let chunked: String = body.as_bytes().chunks(7).map(|c| format!("{:x}\r\n{}\r\n", c.len(), String::from_utf8_lossy(c))).collect();
                write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{chunked}0\r\n\r\n").unwrap();
            }
            seen
        });
        (url, h)
    }

    #[test]
    fn probe_and_incremental_polls() {
        let config = r#"{"title":"Acceleration with g","buffers":[{"name":"accX","size":0},{"name":"accY","size":0},{"name":"accZ","size":0},{"name":"acc_time","size":0}]}"#;
        let first = r#"{"buffer":{"accX":{"buffer":[0.0]},"accY":{"buffer":[0.0]},"accZ":{"buffer":[9.80665]},"acc_time":{"buffer":[10.0]}},"status":{"session":"a1","measuring":true}}"#;
        // two samples since t = 10: |a| = 10.80665 (ENMO 101.97 mg) and 9.80665 (0 mg)
        let second = r#"{"buffer":{"accX":{"buffer":[0.0,0.0]},"accY":{"buffer":[0.0,0.0]},"accZ":{"buffer":[10.80665,9.80665]},"acc_time":{"buffer":[10.5,11.0]}},"status":{"session":"a1","measuring":true}}"#;
        let (url, h) = fake_phyphox(vec![config.into(), first.into(), second.into()]);
        let p = probe(&url).unwrap();
        assert_eq!(p.title, "Acceleration with g");
        let mut st = PhoneState { map: p.map, ..Default::default() };
        let a = poll_phone(&url, &mut st).unwrap();
        assert!(a.activity.abs() < 1e-9 && a.lux.is_nan() && a.measuring == Some(true));
        assert_eq!(st.last_acc_t, Some(10.0));
        let b = poll_phone(&url, &mut st).unwrap();
        assert!((b.activity - 0.5 * 1000.0 / G).abs() < 1e-6, "{}", b.activity);
        let seen = h.join().unwrap();
        assert!(seen[1].starts_with("GET /get?accX&accY&accZ&acc_time "), "{}", seen[1]);
        assert!(seen[2].contains("accX=10%7Cacc_time"), "{}", seen[2]);
    }

    /// Against a real phone: PHYPHOX_URL=http://… cargo test --release --lib real_phone -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_phone() {
        let url = std::env::var("PHYPHOX_URL").expect("set PHYPHOX_URL");
        let p = probe(&url).unwrap();
        println!("experiment: {}\nbuffers: {:?}\nreading: {}", p.title, p.buffers, p.map.describe());
        let mut st = PhoneState { map: p.map, ..Default::default() };
        for i in 0..6 {
            let r = poll_phone(&url, &mut st).unwrap();
            println!("poll {i}: activity {:.2} mg, light {:.3}, measuring {:?}, last light t {:?}", r.activity, r.lux, r.measuring, st.last_lux_t);
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    #[test]
    fn unreachable_phone_and_bad_addresses() {
        // nothing listens on port 1: connection refused, explained
        let e = probe("http://127.0.0.1:1").unwrap_err();
        assert!(e.contains("Cannot reach 127.0.0.1:1") && e.contains("same Wi-Fi"), "{e}");
        assert!(probe("https://192.168.1.50:8080").unwrap_err().contains("http://"));
        assert!(probe("").is_err());
        assert_eq!(split_url("192.168.1.50:8080/").unwrap(), ("192.168.1.50:8080".to_string(), String::new()));
    }

    #[test]
    fn simulated_log_round_trip_and_import() {
        let dir = std::env::temp_dir().join(format!("cosirca_live_test_{}", std::process::id()));
        let path = dir.join("log.csv");
        let _ = std::fs::remove_dir_all(&dir);
        let settings = Settings { url: String::new(), source: Source::Simulated { speed: 3600 }, poll_s: 0.01, log_path: path.clone() };
        let ctx = egui::Context::default();
        let mut r = Recorder::start(settings.clone(), ctx.clone()).unwrap();
        while r.with_status(|s| s.rows.len()) < 60 {
            std::thread::sleep(Duration::from_millis(5));
        }
        r.stop();
        let n = r.with_status(|s| s.rows.len());
        let d = r.dataset();
        assert!(d.channel(Role::Activity).is_some() && d.channel(Role::Light).is_some());
        // resume continues the same file and clock
        let (rows, _) = read_log(&path).unwrap();
        assert_eq!(rows.len(), n);
        let r2 = Recorder::start(settings, ctx).unwrap();
        assert_eq!(r2.with_status(|s| s.resumed), n);
        drop(r2);
        // the log is an ordinary CSV for the importer (and the Python app)
        let p = crate::import::load_path(&path, &Default::default(), crate::import::MergeMode::Auto).unwrap();
        let s = &p.subjects[0];
        assert!(s.channel(Role::Activity).is_some_and(|c| c.unit == "mg") && s.channel(Role::Light).is_some());
        assert!(s.len() >= n);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
