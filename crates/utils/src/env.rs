use colored::Colorize;
use std::{env, sync::OnceLock};

pub fn load_env() {
    static LOAD_ONCE: OnceLock<()> = OnceLock::new();

    LOAD_ONCE.get_or_init(|| {
        if let Err(err) = {
            if let Some(env_file) = env::var_os("ENV_FILE") {
                println!(
                    "{} {}:{} - Loading custom env file from '{}'",
                    "[INFO]".green(),
                    file!(),
                    line!(),
                    env_file.display(),
                );
                dotenvy::from_filename(env_file)
            } else {
                dotenvy::dotenv()
            }
        } {
            println!(
                "{} {}:{} - An error occured while loading the .env file ({})",
                "[WARN]".yellow(),
                file!(),
                line!(),
                err
            )
        }
    });
}
