//! CoSirca: circadian rhythmometry engine and desktop UI.
//!
//! Analysis modules are UI-independent; `app` and `charts` hold the egui front end.

pub mod app;
pub mod charts;
pub mod circular;
pub mod cli;
pub mod cosinor;
pub mod data;
pub mod export;
pub mod import;
pub mod import_ui;
pub mod nonparam;
pub mod pipeline;
pub mod rhythm;
pub mod stats;
pub mod theme;
