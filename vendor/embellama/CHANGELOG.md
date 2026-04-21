# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.1] - 2026-03-27

### Bug Fixes

- Forward TruncateTokens to parallel_tokenize_real in embed_batch ([36c5b37](https://github.com/darjus/embellama/commit/36c5b377ca15ddde22b91c02e68be05e3eb34e83))

## [0.10.0] - 2026-03-21

### Features

- Auto-detect reranker models from GGUF pooling_type metadata ([eac5b81](https://github.com/darjus/embellama/commit/eac5b818644d584eadff41bec0cb2ebc9e5b1b94))

## [0.9.2] - 2026-03-06

### Bug Fixes

- Compile-test README examples as doc tests ([70f7745](https://github.com/darjus/embellama/commit/70f7745f57036b72988accd3a29804f30abfb624))

## [0.9.1] - 2026-03-06

### Bug Fixes

- Fix README version update in release recipe and clean up backend docs ([446662d](https://github.com/darjus/embellama/commit/446662d95e9fc4cb7f8ae20258d1df56605d055c))

## [0.9.0] - 2026-03-05

### Bug Fixes

- Resolve warnings, fix bugs, and strengthen assertions across test suite ([cc9ef0f](https://github.com/darjus/embellama/commit/cc9ef0f3765b32b020f4aff477027ec888ff66b6))
- Eliminate panics, harden error handling, and add security checks ([a8b3c89](https://github.com/darjus/embellama/commit/a8b3c89869e873a7b8c171df81a644fa7748d011))
- Harden reranking with NaN detection, input validation, and correct usage reporting ([d717677](https://github.com/darjus/embellama/commit/d7176777faf23cfab08cd852b2e01807c87fb99b))
- Upgrade sysinfo 0.33 -> 0.38 to fix test_memory_monitor on macOS ([56d3e7d](https://github.com/darjus/embellama/commit/56d3e7d69419a2f04d7d8f3002158d3d4fc37d20))
- Resolve test failures and use test-with for env-gated tests ([79db5d7](https://github.com/darjus/embellama/commit/79db5d782ed78cfaf85e9428c637563f3bdd2315))

### Documentation

- Mention jina-reranker-v2 as supported reranking model ([9963937](https://github.com/darjus/embellama/commit/9963937f001cbec2c022fca2a556139c8c457e03))
- Add git-cliff changelog recipes and documentation ([3f968da](https://github.com/darjus/embellama/commit/3f968da0cfa7d09eea87e43145c49580452c2a8b))

### Features

- Add token truncation support with comprehensive tests ([ef37a73](https://github.com/darjus/embellama/commit/ef37a73b1112fca6004d7110ab3d7f76b72e956f))
- Add PoolingStrategy::None for per-token ColBERT-style embeddings ([ac91cd7](https://github.com/darjus/embellama/commit/ac91cd7908bf97b842e584504bba2e5980a3b4b7))
- Add cross-encoder reranking support with PoolingStrategy::Rank ([ec93890](https://github.com/darjus/embellama/commit/ec9389096f3ab31d73315199d5708b7b68795ef6))
- Upgrade llama-cpp-2 from 0.1.121 to 0.1.138 ([bada0d0](https://github.com/darjus/embellama/commit/bada0d0fdd26c85b3cc9626b8ed93beec12f71b2))

### Testing

- Add comprehensive reranking integration tests ([4745cd8](https://github.com/darjus/embellama/commit/4745cd876685f2168a4aa171abf542dade07fb3d))

## [0.8.0] - 2025-10-28

### Refactoring

- Eliminate ServerConfig duplication with EngineConfig ([a6d7f26](https://github.com/darjus/embellama/commit/a6d7f26a35ed480d3e05e1d1f5bd2ec52bfc276c))

## [0.7.0] - 2025-10-17

### Bug Fixes

- Clear KV cache before embedding generation to prevent contamination ([108656c](https://github.com/darjus/embellama/commit/108656c2ab7e7acf8eefbd2072eb47cb04b63594))
- Remove unsafe logging from Drop implementation to prevent TLS panics ([ef1283f](https://github.com/darjus/embellama/commit/ef1283f8ed8268ad88b4e2c3d606a23ad39d4670))
- Ignore batch ordering for now ([9ad73da](https://github.com/darjus/embellama/commit/9ad73daeb898bfd82744c28ca5a255c3e2db3d33))
- Align normalization with llama-server behavior ([41481ff](https://github.com/darjus/embellama/commit/41481ffc89641d38d46e920aa39caa56117da96c))
- Make test_config_with_backend_detection work without environment variable ([b26ca9f](https://github.com/darjus/embellama/commit/b26ca9fbb17bf3f2ea9c978c6ee2e84d11899875))
- Set bos token to always for embeddings ([7fa03eb](https://github.com/darjus/embellama/commit/7fa03eb07aa78ea2a01eefe74157ef365bab3db6))
- Fix rustdoc ([1d824dc](https://github.com/darjus/embellama/commit/1d824dc4c480ed1436e921bf01e7ab87a638ec70))
- Correct effective_max_tokens to account for per-sequence batching ([5b0049e](https://github.com/darjus/embellama/commit/5b0049e8623c462be952c777f55a8add3299321e))
- Add context_size validation and prevent integer overflow ([7df53c8](https://github.com/darjus/embellama/commit/7df53c8799941350a8676c314fc8701809882099))
- Update justfile tests to run quick property tests, fix server API tests with n_seq_max ([2a63a2b](https://github.com/darjus/embellama/commit/2a63a2b3684f8ee792089302eddb25c9d5df4495))
- Fix doc and small tweak to perf test for CI ([87c4c2f](https://github.com/darjus/embellama/commit/87c4c2ff32b1b6b4a9e8694891c7e888af5f40f9))

### Features

- Add decoder model testing support to justfile ([dd26960](https://github.com/darjus/embellama/commit/dd2696083a9f4df94e481fd137458662e7a40b15))
- Replace gguf crate with custom GGUF metadata reader ([955f6f2](https://github.com/darjus/embellama/commit/955f6f2988e984aa71e2c703f063e3b9bb6d0519))
- Add configurable context size for property tests ([eb07737](https://github.com/darjus/embellama/commit/eb07737ab957a93a6e0d05f940072949fe9f4db4))
- Expose pooling and normalization settings through server CLI ([2cc9697](https://github.com/darjus/embellama/commit/2cc9697110799f658fb5ecdc3e1bc675d949f168))
- Add n_batch parameter for max usable context control ([6fdf866](https://github.com/darjus/embellama/commit/6fdf8663af990df1774b16a58602680289538635))

### Refactoring

- Eliminate code duplication in model.rs ([fd627e3](https://github.com/darjus/embellama/commit/fd627e36e7370b2b7c8b99798140ee803ba69118))
- Remove deprecated methods and fields, updates to use model config ([01b100b](https://github.com/darjus/embellama/commit/01b100bf14570d3bf5e08e396b611f3c7c5ea536))
- Remove BOS token detection logic ([35607d3](https://github.com/darjus/embellama/commit/35607d3e1a3b7fdd54494095d5c8c1fca87d5d6e))
- Unify n_seq_max handling across encoder and decoder models ([4de65f4](https://github.com/darjus/embellama/commit/4de65f4d4b6c06a405b65584eeda548eb56622df))
- Consolidate to single inference worker thread per model ([7f574e2](https://github.com/darjus/embellama/commit/7f574e27395fc0c0fec371d614f3b8570fb3ab46))

## [0.6.1] - 2025-10-08

### Bug Fixes

- Account for embedding output space in batch validation ([951b7c9](https://github.com/darjus/embellama/commit/951b7c9e6fa94001e992518ddbe58eacf42061ee))

## [0.6.0] - 2025-10-07

### Bug Fixes

- Prevent deadlock in TokenCache::evict_oldest ([8706589](https://github.com/darjus/embellama/commit/870658933ea602bd76e1eaa07e3b963174b58ce2))
- Correct memory threshold in test_memory_monitor test ([08e46b8](https://github.com/darjus/embellama/commit/08e46b87b32fbf84d8fa89bf4eea7a8c9519a2a6))
- Adjust test_batch_embeddings_duplicate_texts for n_seq_max=8 ([e45287e](https://github.com/darjus/embellama/commit/e45287e7106b6a6f813f4e7b5368f120e91a1210))
- Use minimum 101 tokens in prefix cache tests ([6a65220](https://github.com/darjus/embellama/commit/6a6522083aeda01384113b95537bfb0e1d9b5c78))

### Features

- Add configurable request timeout and n_seq_max for server ([7d5465e](https://github.com/darjus/embellama/commit/7d5465e31c46f089858fbbce12760d932b1dc61d))

## [0.5.0] - 2025-10-06

### Bug Fixes

- Prevent n_ubatch assertion failures for large inputs ([aace7b9](https://github.com/darjus/embellama/commit/aace7b96c9a51059f0832ddde1c5df142d9971f6))
- Split coverage CI to prevent tarpaulin timeouts ([b4b74e6](https://github.com/darjus/embellama/commit/b4b74e6ca1a6e02669398c557a162bac889bb6a9))
- Ensure context_size field always present in /v1/models API ([1780cb6](https://github.com/darjus/embellama/commit/1780cb67fe96903f955468f1702bd2907c3b7d97))

### Features

- Automate README.md version updates in release script ([a7cd18c](https://github.com/darjus/embellama/commit/a7cd18c94f5677609af18525f7b27e00cabfd4c1))

## [0.4.2] - 2025-10-06

### Features

- Auto-detect context size from GGUF metadata ([0260be0](https://github.com/darjus/embellama/commit/0260be07357cadb6ec4b8770d0475020c257f0df))

## [0.4.1] - 2025-09-26

### Bug Fixes

- Pin llama-cpp-2 to 0.1.121 as 122 crashes. add Jina support as a BERT model ([03db071](https://github.com/darjus/embellama/commit/03db07183038f239ccd009f0b621be2a3e772358))

## [0.4.0] - 2025-09-25

### Bug Fixes

- The Llama.cpp API for setting flash attention in 122, set to auto policy == -1 ([1371064](https://github.com/darjus/embellama/commit/1371064cade571cecb94ce10a440a597bdfe6b0c))

### Features

- Add just release command for automated releases ([a03fc91](https://github.com/darjus/embellama/commit/a03fc91ddabc85559edb82a28bd7416e188c30dc))
- Complete Phase 5 - Advanced KV Cache Optimization ([44e6b5b](https://github.com/darjus/embellama/commit/44e6b5bd79ab22bfb411d4c9842a8acd038d423b))
- Add context size to model API response and fix test compilation ([3990916](https://github.com/darjus/embellama/commit/39909165618532eef1b755a0f0b021a7662ab9b9))

## [0.3.0] - 2025-09-16

### Bug Fixes

- Remove GPU features from docs.rs build configuration ([dad3e9b](https://github.com/darjus/embellama/commit/dad3e9bad772c7b94a378f67331fcf8ad4c413f1))
- Adjust semantic similarity thresholds based on proportional text change ([46f6f69](https://github.com/darjus/embellama/commit/46f6f69bf47f12c20c9960f8ed3b92ee61936f85))

### Features

- Add embedded server support and library interface ([7a4e375](https://github.com/darjus/embellama/commit/7a4e3753f6fef177d8b96caa7651545629d3dc56))

## [0.2.0] - 2025-09-12

### Bug Fixes

- Complete Phase 5 integration test fixes ([f96b479](https://github.com/darjus/embellama/commit/f96b47954ddcee8ed19a9f2fb792af81273e54b8))
- Normalize embeddings and fix error response format for OpenAI compatibility ([20a3299](https://github.com/darjus/embellama/commit/20a329941c7908d96230499036cecc898354a85a))
- Correct test assertion for normalize_embeddings default value ([2b6fdb8](https://github.com/darjus/embellama/commit/2b6fdb82e4c21c414b2d4a96a46c04f85141d368))
- Escape [CLS] in rustdoc comment to fix broken intra-doc link ([7600ef6](https://github.com/darjus/embellama/commit/7600ef664d5e83abfae5a6f7f91b90d0b06aaddf))
- Resolve clippy pedantic warnings and improve code quality ([5749a0e](https://github.com/darjus/embellama/commit/5749a0e3ff1391e867c70309c8513bd631a013ad))
- Resolve property test failures by adding n_ubatch configuration ([b1a21bf](https://github.com/darjus/embellama/commit/b1a21bf7b32ed088f3b4555ebf1321bc0c64603d))
- Add missing CI dependencies for Linux and macOS ([b63dd8b](https://github.com/darjus/embellama/commit/b63dd8be3df09424050e0ee715871511da34d7de))
- Add missing CI dependencies for Linux and macOS ([91cf85f](https://github.com/darjus/embellama/commit/91cf85f1d3770761fc790ccddf3e1a2f4e4adfd8))

### Features

- Implement Phase 1 - server foundation ([59081a7](https://github.com/darjus/embellama/commit/59081a7dead5e1016215dc703b6a07524c93db2e))
- Implement Phase 2 - worker pool architecture ([f7c3a15](https://github.com/darjus/embellama/commit/f7c3a15bc64f94d7e16c439daa7e0fe3c088e587))
- Implement Phase 3 - OpenAI-compatible API endpoints ([a56e8e6](https://github.com/darjus/embellama/commit/a56e8e65ce4571d8c79137629c4f650bfc042f73))
- Implement Phase 4 - Request/Response Pipeline with security fixes ([8e8c3be](https://github.com/darjus/embellama/commit/8e8c3be6f87802de811c4b5bc2a54c242ee51ec9))
- [**breaking**] Add backend feature support for hardware acceleration ([bff6b7c](https://github.com/darjus/embellama/commit/bff6b7c8572205fc1a1fb8141fd9ace0e4ddd6ca))

### Miscellaneous Tasks

- Prepare for v0.1.0 release to crates.io ([9aa780d](https://github.com/darjus/embellama/commit/9aa780d09bac57901293b775a7cc5a70a4811a78))
- Prepare for crates.io publishing ([6b0e2a9](https://github.com/darjus/embellama/commit/6b0e2a9395f87ecec6ac3d622250bddaa41ef0ee))
- Update .gitignore ([51ea9a7](https://github.com/darjus/embellama/commit/51ea9a71bc5eb5478be76f88eacd4ef66e12cf53))
- Cargo fmt ([f51cd8e](https://github.com/darjus/embellama/commit/f51cd8ed3916aead4ddd17e242ebc2d9f3c09c81))
- Cargo clippy -- -W clippy::pedantic --fix ([ca663aa](https://github.com/darjus/embellama/commit/ca663aa8436539ffff2dd0e501e8885512e7d988))
- Update versions ([58668d0](https://github.com/darjus/embellama/commit/58668d0c2be3f648e4cac918fae72543179451c7))
- Clippy fix across features and targets ([6855a5a](https://github.com/darjus/embellama/commit/6855a5a3652a6ea85a4fbe35faf5c731ca93e67c))
- Update deps ([70248d6](https://github.com/darjus/embellama/commit/70248d6fe9aabb41766b8f1f992d5620ccf0e7ba))
- Add pre-commit hooks with uvx/pipx support ([2203b1e](https://github.com/darjus/embellama/commit/2203b1ed32ef344cba6f4473cdd08778c2eb2378))
- Fix fmtcheck ([3abd194](https://github.com/darjus/embellama/commit/3abd1940d1d807b824b426c90751291c758eade7))

### Ci

- Enhance CI/CD pipeline with just commands and model downloads ([52a54e6](https://github.com/darjus/embellama/commit/52a54e65abd01d0a8f3630fa9cde7d4516088ea5))
- Use Ubuntu clang and llvm on linux instead of compiling one ([17f1373](https://github.com/darjus/embellama/commit/17f1373aa9abc799943d67812fc333312daa5d25))

## [0.1.0] - 2025-09-02

### Bug Fixes

- Implement backend singleton to prevent multiple initialization errors ([9ce64ed](https://github.com/darjus/embellama/commit/9ce64ed12ca2291b231221c00cc78d472b9cb96b))

### Documentation

- Separate README.md and DEVELOPMENT.md ([7aa8533](https://github.com/darjus/embellama/commit/7aa853379d5791e6086f0e063218e693aaa164b7))
- Add .clog.toml ([d477333](https://github.com/darjus/embellama/commit/d4773330481af2313f0fbbb06b8632920d2802a5))
- Formatting ([c07dec9](https://github.com/darjus/embellama/commit/c07dec95c253452558ad0838f26335245c57db2a))

### Features

- Implement Phase 1 - Project Setup & Core Infrastructure ([f8ef59c](https://github.com/darjus/embellama/commit/f8ef59c7b9ff60c2557bd3b23e2aae6c3fca675d))
- Implement Phase 2 - Basic Model Management ([6e9f98c](https://github.com/darjus/embellama/commit/6e9f98c83d9275097d28c60ba5764ccf3c54377f))
- Implement Phase 3 - Single Embedding Generation ([47f9839](https://github.com/darjus/embellama/commit/47f98390b1ec505258ea7e987014023fbce17d06))
- Implement Phase 4 - Batch Processing ([3a9988e](https://github.com/darjus/embellama/commit/3a9988e9c889e990870b5b7b98bd33e201444d63))
- Implement Phase 5 - Testing & Documentation ([e92fac2](https://github.com/darjus/embellama/commit/e92fac2c0ac7254248433c0028a50ce96a966373))
- Add comprehensive test infrastructure with real GGUF models ([7dafdbb](https://github.com/darjus/embellama/commit/7dafdbbb76e270bdb22f34302a820f02b9be0477))
- Add n_seq_max configuration and true batch processing ([af0c99f](https://github.com/darjus/embellama/commit/af0c99fa16d530fbd787e697c2460661c1424a9d))

### Miscellaneous Tasks

- Migrate from clog to git-cliff for changelog generation ([1d1066c](https://github.com/darjus/embellama/commit/1d1066ca1d462340818ff8e8436a1d4d586f1170))

### Refactoring

- Move LlamaBackend ownership from model to engine ([08ae8e6](https://github.com/darjus/embellama/commit/08ae8e6fe8ca2c5e983d22e16c81ed961ed316d4))

## [0.0.0] - 2025-08-26

<!-- generated by git-cliff -->
