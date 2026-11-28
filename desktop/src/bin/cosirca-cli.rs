//! Console entry point: `cosirca-cli data.csv --bin-sec 600 --excel report.xlsx`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = cosirca::cli::run(&args) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
