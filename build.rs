use cc;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-search=/home/reid/Documents/coinhsl-install/lib/");
    println!("cargo:rustc-link-lib=dylib=hsl");
    println!("cargo::rerun-if-changed=src/csparse.c");
    // Use the `cc` crate to build a C file and statically link it.
    cc::Build::new()
        .file("src/csparse.c")
        .opt_level(5)
        .compile("csparse");

    println!("cargo:rustc-link-lib=csparse");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .header("src/bindings.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}