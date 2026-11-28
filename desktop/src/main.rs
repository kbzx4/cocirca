#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! CoCirca desktop GUI. Files given on the command line are opened at start-up.
//! The headless runner is the separate `cocirca-cli` binary.

fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("CoCirca — circadian rhythmometry")
            .with_inner_size([1440.0, 920.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native("CoCirca", options, Box::new(|cc| Ok(Box::new(cocirca::app::CocircaApp::new(cc))))) {
        eprintln!("Failed to start the window: {e}");
        std::process::exit(1);
    }
}
