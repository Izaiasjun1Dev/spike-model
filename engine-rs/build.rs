// libmlx.dylib is found next to the binary (how a release ships it), or in
// MLX_SYSTEM_DIR/lib when building against the Python wheel's MLX.
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_SYSTEM_DIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    if let Ok(prefix) = std::env::var("MLX_SYSTEM_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    }
}
