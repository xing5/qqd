// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

fn main() {
    // Validate feature combinations
    validate_backend_features();

    // Print cargo instructions for rerun
    println!("cargo:rerun-if-changed=build.rs");
}

fn validate_backend_features() {
    // Count how many GPU backends are enabled
    let gpu_backends = [
        cfg!(feature = "cuda"),
        cfg!(feature = "metal"),
        cfg!(feature = "vulkan"),
    ];

    let enabled_count = gpu_backends.iter().filter(|&&enabled| enabled).count();

    if enabled_count > 1 {
        // Print warning but don't fail the build
        // This allows users to experiment but warns them about potential issues
        println!("cargo:warning=Multiple GPU backends are enabled. This may cause conflicts.");
        println!("cargo:warning=It's recommended to use only one GPU backend at a time.");
        println!(
            "cargo:warning=Enabled backends: cuda={}, metal={}, vulkan={}",
            cfg!(feature = "cuda"),
            cfg!(feature = "metal"),
            cfg!(feature = "vulkan")
        );
    }

    // Platform-specific warnings
    #[cfg(all(target_os = "macos", feature = "cuda"))]
    println!("cargo:warning=CUDA feature enabled on macOS. CUDA is not supported on macOS.");

    #[cfg(all(target_os = "windows", feature = "metal"))]
    println!("cargo:warning=Metal feature enabled on Windows. Metal is only supported on macOS.");

    #[cfg(all(target_os = "linux", feature = "metal"))]
    println!("cargo:warning=Metal feature enabled on Linux. Metal is only supported on macOS.");

    // Print selected backend for informational purposes
    // Priority order must match detect_best_backend() in src/backend.rs
    if cfg!(all(target_os = "macos", feature = "metal")) {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=metal");
    } else if cfg!(all(not(target_os = "macos"), feature = "cuda")) {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=cuda");
    } else if cfg!(feature = "rocm") {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=rocm");
    } else if cfg!(feature = "vulkan") {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=vulkan");
    } else if cfg!(feature = "openmp") {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=openmp");
    } else {
        println!("cargo:rustc-env=EMBELLAMA_BACKEND=cpu");
    }
}
