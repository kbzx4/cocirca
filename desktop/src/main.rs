#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! CoSirca desktop GUI. Files given on the command line are opened at start-up.
//! The headless runner is the separate `cosirca-cli` binary.

fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("CoSirca — circadian rhythmometry")
            .with_inner_size([1440.0, 920.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native("CoSirca", options, Box::new(|cc| Ok(Box::new(cosirca::app::CosircaApp::new(cc))))) {
        eprintln!("Failed to start the window: {e}");
        std::process::exit(1);
    }
}
