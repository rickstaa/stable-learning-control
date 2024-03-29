# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [6.0.0](https://github.com/rickstaa/stable-learning-control/compare/v5.1.1...v6.0.0) (2024-03-15)


### ⚠ BREAKING CHANGES

* **buffers:** the `ReplayBuffer` and `TrajectoryBuffer` classes don't take a `rew_dim` argument anymore.
* **package:** The package should now be used as `stable_learning_control` instead of `bayesian_learning_control`.
* This package now depends on Gymnasium instead of Gym. Furthermore, it also requires Gymnasium>=26 (see https://gymnasium.farama.org/content/migration-guide/) for more information.
* The `simzoo` package is no longer included and has to be installed separately through the [stable-gym](https://github.com/rickstaa/stable-gym) repository.

### Features

* add docker container ([#379](https://github.com/rickstaa/stable-learning-control/issues/379)) ([972079f](https://github.com/rickstaa/stable-learning-control/commit/972079faf2d87937905c94ff01c766a0c0b2b2cd))
* add hyperparameter variant support for exp config files ([6ea47f0](https://github.com/rickstaa/stable-learning-control/commit/6ea47f0c330e2ea57c98e50165c160ffa43ff53e))
* add small policy test script ([c701d69](https://github.com/rickstaa/stable-learning-control/commit/c701d69c507d735380cbb619037fca4f9a43e8c2))
* add support for dictionary type observation spaces ([e3bf761](https://github.com/rickstaa/stable-learning-control/commit/e3bf76194875576ce6b623a97987696eb7105c0c))
* add torch reproducibility code ([#320](https://github.com/rickstaa/stable-learning-control/issues/320)) ([89ef5a2](https://github.com/rickstaa/stable-learning-control/commit/89ef5a2fb4cd5804b189147e5691db3151a7f368))
* enable GPU device selection ([#406](https://github.com/rickstaa/stable-learning-control/issues/406)) ([73c1374](https://github.com/rickstaa/stable-learning-control/commit/73c1374d9b5cb48487e36cb7bb4f99b1c51c5c3b))
* **exp_cfg:** fix 'start_policy' empty config bug ([e7f3cf9](https://github.com/rickstaa/stable-learning-control/commit/e7f3cf9caaaec690eb934cca85e76b9e201f07fd))
* **exp:** add lambda lr check experiments ([fedfa16](https://github.com/rickstaa/stable-learning-control/commit/fedfa163bf2ff5241664ce008ebc8bc9caab826d))
* **exp:** add min statistics to rstaa2024 data analysis ([16ff564](https://github.com/rickstaa/stable-learning-control/commit/16ff56433977b11a5bb92e29796c08cf2fc187a9))
* **exp:** add rstaa 2024 data analysis script ([f0df0a4](https://github.com/rickstaa/stable-learning-control/commit/f0df0a4720bba5625274f5963fa715ea0f200ba0))
* **exp:** rename CompOscillator lambda experiment ([ec7103c](https://github.com/rickstaa/stable-learning-control/commit/ec7103c3c656aea14d86433157e79040754edd21))
* improve 'eval_robustness' utility ([#313](https://github.com/rickstaa/stable-learning-control/issues/313)) ([3985867](https://github.com/rickstaa/stable-learning-control/commit/3985867fb1d452a343c7abcccda7ae37c74286dd))
* improve hyperparmeter tuning, logger and add W&B logging ([#314](https://github.com/rickstaa/stable-learning-control/issues/314)) ([74afd65](https://github.com/rickstaa/stable-learning-control/commit/74afd658bcd22579221b5f51c9e5d074f1929857))
* **lac:** add finite-horizon Lyapunov Candidate ([#328](https://github.com/rickstaa/stable-learning-control/issues/328)) ([ed2c85d](https://github.com/rickstaa/stable-learning-control/commit/ed2c85dcb9653aa8253e40870842dbefc5b33cb4))
* **lac:** implement Han et al. 2020 hyperparameters ([#399](https://github.com/rickstaa/stable-learning-control/issues/399)) ([574b651](https://github.com/rickstaa/stable-learning-control/commit/574b6516cc599920a44286e917f730b405b6dbe9))
* **latc:** add LATC algorithm ([#323](https://github.com/rickstaa/stable-learning-control/issues/323)) ([d74c64f](https://github.com/rickstaa/stable-learning-control/commit/d74c64f76b8aa1d10b199efcd9ea329f79bad547))
* **package:** renames the package to `stable-learning-control` ([#257](https://github.com/rickstaa/stable-learning-control/issues/257)) ([1133d0a](https://github.com/rickstaa/stable-learning-control/commit/1133d0a2c9294a85cda592ed46627a4aba7157bc))
* **plot:** improve plotter arguments ([#425](https://github.com/rickstaa/stable-learning-control/issues/425)) ([c7202a2](https://github.com/rickstaa/stable-learning-control/commit/c7202a2d3913b50833325bf180e2a4a18c79c91f))
* **pytorch:** add alpha/lambda learning rate customization ([#412](https://github.com/rickstaa/stable-learning-control/issues/412)) ([6feb749](https://github.com/rickstaa/stable-learning-control/commit/6feb749dfcbb9926c69f48f00640434b057ce962))
* replace OpenAi gym with Gymnasium ([#255](https://github.com/rickstaa/stable-learning-control/issues/255)) ([9873a03](https://github.com/rickstaa/stable-learning-control/commit/9873a0311511c6e9e08ea0957bb008e8a7e6f109))
* **tf2:** add alpha/lambda learning rate customization ([#416](https://github.com/rickstaa/stable-learning-control/issues/416)) ([712e94b](https://github.com/rickstaa/stable-learning-control/commit/712e94b4f9b59db46137acdd413596de2ed0a629))
* **wandb:** add 'wandb_run_name' argument ([#325](https://github.com/rickstaa/stable-learning-control/issues/325)) ([e0a0b9d](https://github.com/rickstaa/stable-learning-control/commit/e0a0b9d43277bec384f3d82d6d80d38a0d20ba5c))


### Bug Fixes

* Address flake8 and black formatting issues ([#395](https://github.com/rickstaa/stable-learning-control/issues/395)) ([517ee30](https://github.com/rickstaa/stable-learning-control/commit/517ee307afd605bcbb5c92a4e49c348666a8cb77))
* correct finite horizon buffer calculation ([#398](https://github.com/rickstaa/stable-learning-control/issues/398)) ([779201c](https://github.com/rickstaa/stable-learning-control/commit/779201c03e35e5125aaca975b6790f034f912c3c))
* correctly close gymnasium environments ([#340](https://github.com/rickstaa/stable-learning-control/issues/340)) ([a179176](https://github.com/rickstaa/stable-learning-control/commit/a1791761737c0f83220461b036c55e0acbc68cf5))
* ensure 'test_policy' works with gymnasium&gt;=0.28.1 ([#276](https://github.com/rickstaa/stable-learning-control/issues/276)) ([80fe370](https://github.com/rickstaa/stable-learning-control/commit/80fe370a6132e1663fd2dadb4d8397a5adea1b3b))
* fix 'test_policy' episode length bug ([#292](https://github.com/rickstaa/stable-learning-control/issues/292)) ([6d34f74](https://github.com/rickstaa/stable-learning-control/commit/6d34f74f8a18104ad92644aa624cee2c57135aad))
* fix 'test_policy' rendering ([#291](https://github.com/rickstaa/stable-learning-control/issues/291)) ([48443ca](https://github.com/rickstaa/stable-learning-control/commit/48443ca1b15854b00f67272ab88a2766624d5530))
* fix environment validation ([#280](https://github.com/rickstaa/stable-learning-control/issues/280)) ([a8a0346](https://github.com/rickstaa/stable-learning-control/commit/a8a0346c9e2f4bf583a15dea11bdac00b7f9db4d))
* fix plot dataframe loading ([#338](https://github.com/rickstaa/stable-learning-control/issues/338)) ([90e16e9](https://github.com/rickstaa/stable-learning-control/commit/90e16e9385df8f1eac4a3626aebaa56ebd6e7b61))
* fix several env/policy load bugs ([dddd4d8](https://github.com/rickstaa/stable-learning-control/commit/dddd4d8343301e322c82285b777a259bb6221cb0))
* fix several policy loading problems ([51a664e](https://github.com/rickstaa/stable-learning-control/commit/51a664e3d18a3b1c315d943f328be66ed5ab683a))
* improve tensorflow lazy import ([#272](https://github.com/rickstaa/stable-learning-control/issues/272)) ([75192a4](https://github.com/rickstaa/stable-learning-control/commit/75192a4bb456f7bbb0391fb1e63e4622054d2e33))
* **pytorch:** correct epoch-based learning rate decay behavior ([#410](https://github.com/rickstaa/stable-learning-control/issues/410)) ([a8df90f](https://github.com/rickstaa/stable-learning-control/commit/a8df90f1b0dafb7f01c76767e21bbb3075bfeff3))
* **pytorch:** correct step-based learning rate decay ([#405](https://github.com/rickstaa/stable-learning-control/issues/405)) ([7d7ac76](https://github.com/rickstaa/stable-learning-control/commit/7d7ac76b561d82d9ebbcc926476d5db4862f711e))
* **pytorch:** ensure correct application of constant learning rate ([#411](https://github.com/rickstaa/stable-learning-control/issues/411)) ([2b3693e](https://github.com/rickstaa/stable-learning-control/commit/2b3693e0baa3d2656a42c859de3e4d2e922d4374))
* **pytorch:** fix learning rate decay defaults ([#414](https://github.com/rickstaa/stable-learning-control/issues/414)) ([27964fe](https://github.com/rickstaa/stable-learning-control/commit/27964fe46b18175c70ed16186e16683c4eeb1afb))
* **pytorch:** resolve critical action rescaling bug ([#403](https://github.com/rickstaa/stable-learning-control/issues/403)) ([71d4f64](https://github.com/rickstaa/stable-learning-control/commit/71d4f643bdceb0e7ca26e158ea035327823c1557))
* remove 'simzoo' submodule ([#246](https://github.com/rickstaa/stable-learning-control/issues/246)) ([0122aae](https://github.com/rickstaa/stable-learning-control/commit/0122aaeb79d6c62da48591199e3a4026043c628b))
* Resolve ruamel safe_load deprecation issue ([#396](https://github.com/rickstaa/stable-learning-control/issues/396)) ([cfcf81c](https://github.com/rickstaa/stable-learning-control/commit/cfcf81cf01b4b3a89edd746990594552255cd42f))
* **run:** resolve issue with data_dir input argument ([#409](https://github.com/rickstaa/stable-learning-control/issues/409)) ([8d93610](https://github.com/rickstaa/stable-learning-control/commit/8d936101a280c068dfcf61c1d7919453ce1a0b3c))
* **tf2:** correct off-by-one error in learning rate decay calculation ([#415](https://github.com/rickstaa/stable-learning-control/issues/415)) ([6ab5001](https://github.com/rickstaa/stable-learning-control/commit/6ab5001c588a97ec4379aed3c224d785964448c1))
* **tf2:** correct step-based learning rate decay ([#407](https://github.com/rickstaa/stable-learning-control/issues/407)) ([642a193](https://github.com/rickstaa/stable-learning-control/commit/642a19330d70e53cba845c9e10916b76332beec8))
* **tf2:** fix critical tf2 gradient update bug ([#322](https://github.com/rickstaa/stable-learning-control/issues/322)) ([dfc239b](https://github.com/rickstaa/stable-learning-control/commit/dfc239b53f4b3de084354a5411ff99b0207d6cbf))
* **torch:** handle 'update_after' set to zero ([#408](https://github.com/rickstaa/stable-learning-control/issues/408)) ([7999590](https://github.com/rickstaa/stable-learning-control/commit/7999590da2f65579343fd26600ef6ab60b5fe171))
* **wandb:** fix wandb config format and run name ([#317](https://github.com/rickstaa/stable-learning-control/issues/317)) ([ca048de](https://github.com/rickstaa/stable-learning-control/commit/ca048deff30e33f102d48006abbc1a6153754c6a))


### Documentation

* add docker ownership admonition ([58a8e70](https://github.com/rickstaa/stable-learning-control/commit/58a8e702d218970c9294d54f5ef7e9f526b3d94c))
* add policy test utility to docker page ([5d39316](https://github.com/rickstaa/stable-learning-control/commit/5d3931619e24558277b378f7aabb2bbf2bb3a025))
* add pre-commit note ([#341](https://github.com/rickstaa/stable-learning-control/issues/341)) ([9853d05](https://github.com/rickstaa/stable-learning-control/commit/9853d05a8d5395eab900837a44a38d0e183ead77))
* add ros-gazebo-gym positive reward comment ([#349](https://github.com/rickstaa/stable-learning-control/issues/349)) ([9f5c434](https://github.com/rickstaa/stable-learning-control/commit/9f5c4340732911138249bc0de57edce31472bff9))
* add snapshots comment to CONTRIBUTING ([c2031b2](https://github.com/rickstaa/stable-learning-control/commit/c2031b23eb0b510dbb464a2453f1524f47e08099))
* add zenodo reference information ([#300](https://github.com/rickstaa/stable-learning-control/issues/300)) ([7180fb3](https://github.com/rickstaa/stable-learning-control/commit/7180fb37f3027338873ad487c0fbeaaaeec114e5))
* **contributing:** updates the contributing documentation ([bac8787](https://github.com/rickstaa/stable-learning-control/commit/bac8787b5edd34fcf5d1f418ee9811db49340db5))
* fix broken external links ([51317c0](https://github.com/rickstaa/stable-learning-control/commit/51317c0acda779d12ed2096cbda6f65e83e22874))
* fix contribution links ([#245](https://github.com/rickstaa/stable-learning-control/issues/245)) ([cccc5a7](https://github.com/rickstaa/stable-learning-control/commit/cccc5a7d4bc6239d90b60c1b66e0b53e89e49987))
* fix doc code highlights ([#404](https://github.com/rickstaa/stable-learning-control/issues/404)) ([99126f7](https://github.com/rickstaa/stable-learning-control/commit/99126f7d58c4d68f6cf56d95e38e566b85101ad5))
* fix documentation ([#377](https://github.com/rickstaa/stable-learning-control/issues/377)) ([c3b47c5](https://github.com/rickstaa/stable-learning-control/commit/c3b47c51ce6770858e6f33a1b0de7bf5c9161c06))
* fix documentation latex formulas ([#383](https://github.com/rickstaa/stable-learning-control/issues/383)) ([#385](https://github.com/rickstaa/stable-learning-control/issues/385)) ([582568b](https://github.com/rickstaa/stable-learning-control/commit/582568b29e9747d174ebe5204c101c3602c924de))
* fix incorrect package naming ([#282](https://github.com/rickstaa/stable-learning-control/issues/282)) ([5337841](https://github.com/rickstaa/stable-learning-control/commit/533784112dac20adbd649d0b2a81ccbbf41119de))
* fix incorrect run command ([43df73f](https://github.com/rickstaa/stable-learning-control/commit/43df73f24d0ae4ae21b868b5c4bca336e5017db2))
* fix links ([#268](https://github.com/rickstaa/stable-learning-control/issues/268)) ([21c5bcf](https://github.com/rickstaa/stable-learning-control/commit/21c5bcf8823daf88d130654297383cd2a4ad14f2))
* fix SAC alg run command ([#380](https://github.com/rickstaa/stable-learning-control/issues/380)) ([cf0a7a3](https://github.com/rickstaa/stable-learning-control/commit/cf0a7a3eceb4434bba32be653f70a0d3e331ef47))
* fix typos ([30df6cb](https://github.com/rickstaa/stable-learning-control/commit/30df6cb7fca0269f9cc336d69d56193bfb40c541))
* improve code API documentation ([#306](https://github.com/rickstaa/stable-learning-control/issues/306)) ([16d5e26](https://github.com/rickstaa/stable-learning-control/commit/16d5e260b212f6fea29d2e80d14e96474c7001f0))
* improve docs ([#344](https://github.com/rickstaa/stable-learning-control/issues/344)) ([908fd78](https://github.com/rickstaa/stable-learning-control/commit/908fd78886676d55a7e2f760b62bb280e915f124))
* improve documentation and package.json ([#331](https://github.com/rickstaa/stable-learning-control/issues/331)) ([e9fb90e](https://github.com/rickstaa/stable-learning-control/commit/e9fb90e2088401336a066684b26ecb81beb2d43e))
* improve documentation defaults ([be99008](https://github.com/rickstaa/stable-learning-control/commit/be990083f951c8cb12ee30948d12cb3db2864913))
* improve documentation urls and syntax ([#346](https://github.com/rickstaa/stable-learning-control/issues/346)) ([22037cb](https://github.com/rickstaa/stable-learning-control/commit/22037cbb27eed57ca8ee196df242a422c2643e38))
* improve package description ([#260](https://github.com/rickstaa/stable-learning-control/issues/260)) ([eabfef3](https://github.com/rickstaa/stable-learning-control/commit/eabfef3bdb3ba26de988de4ebce383024f8865df))
* improve polyak description ([#417](https://github.com/rickstaa/stable-learning-control/issues/417)) ([2404a4e](https://github.com/rickstaa/stable-learning-control/commit/2404a4e78946cc222bbabae630ec40af3f63953d))
* improve the documentation ([#265](https://github.com/rickstaa/stable-learning-control/issues/265)) ([829eb05](https://github.com/rickstaa/stable-learning-control/commit/829eb05cb8043d2612a2225045c5686259570e61))
* incorporate Docker usage instructions ([#401](https://github.com/rickstaa/stable-learning-control/issues/401)) ([2ba8031](https://github.com/rickstaa/stable-learning-control/commit/2ba80311798002d5278a0ed421cdf19737170d7b))
* **sphinx:** fixes documentation errors ([38f2da0](https://github.com/rickstaa/stable-learning-control/commit/38f2da0a041026c374db5f50be1eac12fad32a20))
* **sphinx:** updates documentation installation instructions ([ddadeb6](https://github.com/rickstaa/stable-learning-control/commit/ddadeb626f9e69c532370553d9e3f0b444a3f832))
* update admonitions to new GFM specs ([ae7d3bc](https://github.com/rickstaa/stable-learning-control/commit/ae7d3bc652ee399c648cab8fca7fc68d707da6d0))
* update contribution note ([#316](https://github.com/rickstaa/stable-learning-control/issues/316)) ([44a35af](https://github.com/rickstaa/stable-learning-control/commit/44a35af889331a1b1077a5d8035ec78ef084b368))
* update docs make file ([6eeb56e](https://github.com/rickstaa/stable-learning-control/commit/6eeb56e52ff7cb7086b46f6af91a035bebe9c959))
* update documentation ([7ceee46](https://github.com/rickstaa/stable-learning-control/commit/7ceee46453e201ff3dc106ca5f879a8a90cc8ad7))
* update documentation and docstrings ([5548505](https://github.com/rickstaa/stable-learning-control/commit/55485055df074f12918d59add178a353d1ca895f))
* update numpy intersphinx link ([581333d](https://github.com/rickstaa/stable-learning-control/commit/581333d40d8df3836b0f51277b141194273c5c47))
* update sandbox description ([#329](https://github.com/rickstaa/stable-learning-control/issues/329)) ([f60cc3b](https://github.com/rickstaa/stable-learning-control/commit/f60cc3b1f76fa78cc5efc81621f977f86bade5cb))
* updates code guidelines documentation ([4f96a46](https://github.com/rickstaa/stable-learning-control/commit/4f96a460a5a85972de1b38538a526945eda1dd5f))
* updates docs ci readme badge ([074b369](https://github.com/rickstaa/stable-learning-control/commit/074b3697f10256997a6a6bb1a62a41212c8ba239))


### Code Refactoring

* **buffers:** remove unused 'rew_dim' argument ([#327](https://github.com/rickstaa/stable-learning-control/issues/327)) ([a69a7f6](https://github.com/rickstaa/stable-learning-control/commit/a69a7f6f3c8c24b1874d6039ed2f798b2c931c0d))

## [5.1.1](https://github.com/rickstaa/stable-learning-control/compare/v5.1.0...v5.1.1) (2024-01-23)


### Documentation

* fix documentation latex formulas ([#383](https://github.com/rickstaa/stable-learning-control/issues/383)) ([#385](https://github.com/rickstaa/stable-learning-control/issues/385)) ([582568b](https://github.com/rickstaa/stable-learning-control/commit/582568b29e9747d174ebe5204c101c3602c924de))

## [5.1.0](https://github.com/rickstaa/stable-learning-control/compare/v5.0.5...v5.1.0) (2024-01-23)


### Features

* add docker container ([#379](https://github.com/rickstaa/stable-learning-control/issues/379)) ([972079f](https://github.com/rickstaa/stable-learning-control/commit/972079faf2d87937905c94ff01c766a0c0b2b2cd))


### Documentation

* fix documentation ([#377](https://github.com/rickstaa/stable-learning-control/issues/377)) ([c3b47c5](https://github.com/rickstaa/stable-learning-control/commit/c3b47c51ce6770858e6f33a1b0de7bf5c9161c06))
* fix SAC alg run command ([#380](https://github.com/rickstaa/stable-learning-control/issues/380)) ([cf0a7a3](https://github.com/rickstaa/stable-learning-control/commit/cf0a7a3eceb4434bba32be653f70a0d3e331ef47))

## [5.0.5](https://github.com/rickstaa/stable-learning-control/compare/v5.0.4...v5.0.5) (2023-08-31)


### Documentation

* add ros-gazebo-gym positive reward comment ([#349](https://github.com/rickstaa/stable-learning-control/issues/349)) ([9f5c434](https://github.com/rickstaa/stable-learning-control/commit/9f5c4340732911138249bc0de57edce31472bff9))

## [5.0.4](https://github.com/rickstaa/stable-learning-control/compare/v5.0.3...v5.0.4) (2023-08-31)


### Documentation

* improve documentation urls and syntax ([#346](https://github.com/rickstaa/stable-learning-control/issues/346)) ([22037cb](https://github.com/rickstaa/stable-learning-control/commit/22037cbb27eed57ca8ee196df242a422c2643e38))

## [5.0.3](https://github.com/rickstaa/stable-learning-control/compare/v5.0.2...v5.0.3) (2023-08-29)


### Documentation

* improve docs ([#344](https://github.com/rickstaa/stable-learning-control/issues/344)) ([908fd78](https://github.com/rickstaa/stable-learning-control/commit/908fd78886676d55a7e2f760b62bb280e915f124))

## [5.0.2](https://github.com/rickstaa/stable-learning-control/compare/v5.0.1...v5.0.2) (2023-08-28)


### Bug Fixes

* correctly close gymnasium environments ([#340](https://github.com/rickstaa/stable-learning-control/issues/340)) ([a179176](https://github.com/rickstaa/stable-learning-control/commit/a1791761737c0f83220461b036c55e0acbc68cf5))
* fix plot dataframe loading ([#338](https://github.com/rickstaa/stable-learning-control/issues/338)) ([90e16e9](https://github.com/rickstaa/stable-learning-control/commit/90e16e9385df8f1eac4a3626aebaa56ebd6e7b61))


### Documentation

* add pre-commit note ([#341](https://github.com/rickstaa/stable-learning-control/issues/341)) ([9853d05](https://github.com/rickstaa/stable-learning-control/commit/9853d05a8d5395eab900837a44a38d0e183ead77))

## [5.0.1](https://github.com/rickstaa/stable-learning-control/compare/v5.0.0...v5.0.1) (2023-08-14)


### Documentation

* improve documentation and package.json ([#331](https://github.com/rickstaa/stable-learning-control/issues/331)) ([e9fb90e](https://github.com/rickstaa/stable-learning-control/commit/e9fb90e2088401336a066684b26ecb81beb2d43e))
* update sandbox description ([#329](https://github.com/rickstaa/stable-learning-control/issues/329)) ([f60cc3b](https://github.com/rickstaa/stable-learning-control/commit/f60cc3b1f76fa78cc5efc81621f977f86bade5cb))

## [5.0.0](https://github.com/rickstaa/stable-learning-control/compare/v4.3.0...v5.0.0) (2023-08-12)


### ⚠ BREAKING CHANGES

* **buffers:** the `ReplayBuffer` and `TrajectoryBuffer` classes don't take a `rew_dim` argument anymore.

### Features

* **lac:** add finite-horizon Lyapunov Candidate ([#328](https://github.com/rickstaa/stable-learning-control/issues/328)) ([ed2c85d](https://github.com/rickstaa/stable-learning-control/commit/ed2c85dcb9653aa8253e40870842dbefc5b33cb4))
* **wandb:** add 'wandb_run_name' argument ([#325](https://github.com/rickstaa/stable-learning-control/issues/325)) ([e0a0b9d](https://github.com/rickstaa/stable-learning-control/commit/e0a0b9d43277bec384f3d82d6d80d38a0d20ba5c))


### Code Refactoring

* **buffers:** remove unused 'rew_dim' argument ([#327](https://github.com/rickstaa/stable-learning-control/issues/327)) ([a69a7f6](https://github.com/rickstaa/stable-learning-control/commit/a69a7f6f3c8c24b1874d6039ed2f798b2c931c0d))

## [4.3.0](https://github.com/rickstaa/stable-learning-control/compare/v4.2.0...v4.3.0) (2023-08-11)


### Features

* add torch reproducibility code ([#320](https://github.com/rickstaa/stable-learning-control/issues/320)) ([89ef5a2](https://github.com/rickstaa/stable-learning-control/commit/89ef5a2fb4cd5804b189147e5691db3151a7f368))
* **latc:** add LATC algorithm ([#323](https://github.com/rickstaa/stable-learning-control/issues/323)) ([d74c64f](https://github.com/rickstaa/stable-learning-control/commit/d74c64f76b8aa1d10b199efcd9ea329f79bad547))


### Bug Fixes

* **tf2:** fix critical tf2 gradient update bug ([#322](https://github.com/rickstaa/stable-learning-control/issues/322)) ([dfc239b](https://github.com/rickstaa/stable-learning-control/commit/dfc239b53f4b3de084354a5411ff99b0207d6cbf))

## [4.2.0](https://github.com/rickstaa/stable-learning-control/compare/v4.1.0...v4.2.0) (2023-08-08)


### Features

* improve hyperparmeter tuning, logger and add W&B logging ([#314](https://github.com/rickstaa/stable-learning-control/issues/314)) ([74afd65](https://github.com/rickstaa/stable-learning-control/commit/74afd658bcd22579221b5f51c9e5d074f1929857))


### Documentation

* update contribution note ([#316](https://github.com/rickstaa/stable-learning-control/issues/316)) ([44a35af](https://github.com/rickstaa/stable-learning-control/commit/44a35af889331a1b1077a5d8035ec78ef084b368))

## [4.1.0](https://github.com/rickstaa/stable-learning-control/compare/v4.0.10...v4.1.0) (2023-08-02)


### Features

* improve 'eval_robustness' utility ([#313](https://github.com/rickstaa/stable-learning-control/issues/313)) ([3985867](https://github.com/rickstaa/stable-learning-control/commit/3985867fb1d452a343c7abcccda7ae37c74286dd))


### Documentation

* update admonitions to new GFM specs ([ae7d3bc](https://github.com/rickstaa/stable-learning-control/commit/ae7d3bc652ee399c648cab8fca7fc68d707da6d0))

## [4.0.10](https://github.com/rickstaa/stable-learning-control/compare/v4.0.9...v4.0.10) (2023-07-15)


### Documentation

* improve code API documentation ([#306](https://github.com/rickstaa/stable-learning-control/issues/306)) ([16d5e26](https://github.com/rickstaa/stable-learning-control/commit/16d5e260b212f6fea29d2e80d14e96474c7001f0))

## [4.0.9](https://github.com/rickstaa/stable-learning-control/compare/v4.0.8...v4.0.9) (2023-07-11)


### Documentation

* add zenodo reference information ([#300](https://github.com/rickstaa/stable-learning-control/issues/300)) ([7180fb3](https://github.com/rickstaa/stable-learning-control/commit/7180fb37f3027338873ad487c0fbeaaaeec114e5))

## [4.0.8](https://github.com/rickstaa/stable-learning-control/compare/v4.0.7...v4.0.8) (2023-07-10)


### Documentation

* add snapshots comment to CONTRIBUTING ([c2031b2](https://github.com/rickstaa/stable-learning-control/commit/c2031b23eb0b510dbb464a2453f1524f47e08099))

## [4.0.7](https://github.com/rickstaa/stable-learning-control/compare/v4.0.6...v4.0.7) (2023-07-09)


### Bug Fixes

* fix 'test_policy' episode length bug ([#292](https://github.com/rickstaa/stable-learning-control/issues/292)) ([6d34f74](https://github.com/rickstaa/stable-learning-control/commit/6d34f74f8a18104ad92644aa624cee2c57135aad))

## [4.0.6](https://github.com/rickstaa/stable-learning-control/compare/v4.0.5...v4.0.6) (2023-07-09)


### Bug Fixes

* fix 'test_policy' rendering ([#291](https://github.com/rickstaa/stable-learning-control/issues/291)) ([48443ca](https://github.com/rickstaa/stable-learning-control/commit/48443ca1b15854b00f67272ab88a2766624d5530))


### Documentation

* improve documentation defaults ([be99008](https://github.com/rickstaa/stable-learning-control/commit/be990083f951c8cb12ee30948d12cb3db2864913))

## [4.0.5](https://github.com/rickstaa/stable-learning-control/compare/v4.0.4...v4.0.5) (2023-07-06)


### Bug Fixes

* ensure 'test_policy' works with gymnasium&gt;=0.28.1 ([#276](https://github.com/rickstaa/stable-learning-control/issues/276)) ([80fe370](https://github.com/rickstaa/stable-learning-control/commit/80fe370a6132e1663fd2dadb4d8397a5adea1b3b))
* fix environment validation ([#280](https://github.com/rickstaa/stable-learning-control/issues/280)) ([a8a0346](https://github.com/rickstaa/stable-learning-control/commit/a8a0346c9e2f4bf583a15dea11bdac00b7f9db4d))


### Documentation

* fix incorrect package naming ([#282](https://github.com/rickstaa/stable-learning-control/issues/282)) ([5337841](https://github.com/rickstaa/stable-learning-control/commit/533784112dac20adbd649d0b2a81ccbbf41119de))

## [4.0.4](https://github.com/rickstaa/stable-learning-control/compare/v4.0.3...v4.0.4) (2023-07-01)


### Bug Fixes

* improve tensorflow lazy import ([#272](https://github.com/rickstaa/stable-learning-control/issues/272)) ([75192a4](https://github.com/rickstaa/stable-learning-control/commit/75192a4bb456f7bbb0391fb1e63e4622054d2e33))

## [4.0.3](https://github.com/rickstaa/stable-learning-control/compare/v4.0.2...v4.0.3) (2023-06-29)


### Documentation

* fix links ([#268](https://github.com/rickstaa/stable-learning-control/issues/268)) ([21c5bcf](https://github.com/rickstaa/stable-learning-control/commit/21c5bcf8823daf88d130654297383cd2a4ad14f2))

## [4.0.2](https://github.com/rickstaa/stable-learning-control/compare/v4.0.1...v4.0.2) (2023-06-29)


### Documentation

* improve the documentation ([#265](https://github.com/rickstaa/stable-learning-control/issues/265)) ([829eb05](https://github.com/rickstaa/stable-learning-control/commit/829eb05cb8043d2612a2225045c5686259570e61))

## [4.0.1](https://github.com/rickstaa/stable-learning-control/compare/v4.0.0...v4.0.1) (2023-06-22)


### Documentation

* improve package description ([#260](https://github.com/rickstaa/stable-learning-control/issues/260)) ([eabfef3](https://github.com/rickstaa/stable-learning-control/commit/eabfef3bdb3ba26de988de4ebce383024f8865df))

## [4.0.0](https://github.com/rickstaa/stable-learning-control/compare/v3.0.0...v4.0.0) (2023-06-22)


### ⚠ BREAKING CHANGES

* **package:** The package should now be used as `stable_learning_control` instead of `bayesian_learning_control`.
* This package now depends on Gymnasium instead of Gym. Furthermore, it also requires Gymnasium>=26 (see https://gymnasium.farama.org/content/migration-guide/) for more information.

### Features

* **package:** renames the package to `stable-learning-control` ([#257](https://github.com/rickstaa/stable-learning-control/issues/257)) ([1133d0a](https://github.com/rickstaa/stable-learning-control/commit/1133d0a2c9294a85cda592ed46627a4aba7157bc))
* replace OpenAi gym with Gymnasium ([#255](https://github.com/rickstaa/stable-learning-control/issues/255)) ([9873a03](https://github.com/rickstaa/stable-learning-control/commit/9873a0311511c6e9e08ea0957bb008e8a7e6f109))

## [3.0.0](https://github.com/rickstaa/stable-learning-control/compare/v2.0.0...v3.0.0) (2023-06-21)


### ⚠ BREAKING CHANGES

* The `simzoo` package is no longer included and has to be installed separately through the [stable-gym](https://github.com/rickstaa/stable-gym) repository.

### Features

* add hyperparameter variant support for exp config files ([6ea47f0](https://github.com/rickstaa/stable-learning-control/commit/6ea47f0c330e2ea57c98e50165c160ffa43ff53e))
* add small policy test script ([c701d69](https://github.com/rickstaa/stable-learning-control/commit/c701d69c507d735380cbb619037fca4f9a43e8c2))
* add support for dictionary type observation spaces ([e3bf761](https://github.com/rickstaa/stable-learning-control/commit/e3bf76194875576ce6b623a97987696eb7105c0c))


### Bug Fixes

* fix several env/policy load bugs ([dddd4d8](https://github.com/rickstaa/stable-learning-control/commit/dddd4d8343301e322c82285b777a259bb6221cb0))
* fix several policy loading problems ([51a664e](https://github.com/rickstaa/stable-learning-control/commit/51a664e3d18a3b1c315d943f328be66ed5ab683a))
* fixes npm release action ([e47ef0f](https://github.com/rickstaa/stable-learning-control/commit/e47ef0fc3936496f5dfed0515b776812244dfcd6))
* **package.json:** this commit makes sure the release run script contains no bugs ([ef958ca](https://github.com/rickstaa/stable-learning-control/commit/ef958cac744bc47202c39f4addef25d9d142c1cd))
* remove 'simzoo' submodule ([#246](https://github.com/rickstaa/stable-learning-control/issues/246)) ([0122aae](https://github.com/rickstaa/stable-learning-control/commit/0122aaeb79d6c62da48591199e3a4026043c628b))


### Documentation

* **contributing:** updates contributing guide ([ce126c4](https://github.com/rickstaa/stable-learning-control/commit/ce126c4969b0bd8903451576338b63d146917219))
* **contributing:** updates the contributing documentation ([bac8787](https://github.com/rickstaa/stable-learning-control/commit/bac8787b5edd34fcf5d1f418ee9811db49340db5))
* fix broken external links ([51317c0](https://github.com/rickstaa/stable-learning-control/commit/51317c0acda779d12ed2096cbda6f65e83e22874))
* fix contribution links ([#245](https://github.com/rickstaa/stable-learning-control/issues/245)) ([cccc5a7](https://github.com/rickstaa/stable-learning-control/commit/cccc5a7d4bc6239d90b60c1b66e0b53e89e49987))
* fix typos ([30df6cb](https://github.com/rickstaa/stable-learning-control/commit/30df6cb7fca0269f9cc336d69d56193bfb40c541))
* **readme.md:** adds contributing warning to readme ([d848a14](https://github.com/rickstaa/stable-learning-control/commit/d848a14db999a829452484b33e3579af8c4e4544))
* **sphinx:** fixes documentation errors ([38f2da0](https://github.com/rickstaa/stable-learning-control/commit/38f2da0a041026c374db5f50be1eac12fad32a20))
* **sphinx:** updates documentation installation instructions ([ddadeb6](https://github.com/rickstaa/stable-learning-control/commit/ddadeb626f9e69c532370553d9e3f0b444a3f832))
* update docs make file ([6eeb56e](https://github.com/rickstaa/stable-learning-control/commit/6eeb56e52ff7cb7086b46f6af91a035bebe9c959))
* update documentation ([7ceee46](https://github.com/rickstaa/stable-learning-control/commit/7ceee46453e201ff3dc106ca5f879a8a90cc8ad7))
* update documentation and docstrings ([5548505](https://github.com/rickstaa/stable-learning-control/commit/55485055df074f12918d59add178a353d1ca895f))
* update numpy intersphinx link ([581333d](https://github.com/rickstaa/stable-learning-control/commit/581333d40d8df3836b0f51277b141194273c5c47))
* updates code guidelines documentation ([4f96a46](https://github.com/rickstaa/stable-learning-control/commit/4f96a460a5a85972de1b38538a526945eda1dd5f))
* updates docs ci readme badge ([074b369](https://github.com/rickstaa/stable-learning-control/commit/074b3697f10256997a6a6bb1a62a41212c8ba239))

## [2.0.0](https://github.com/rickstaa/stable-learning-control/compare/stable-learning-control-v1.3.73...stable-learning-control-v2.0.0) (2023-06-21)


### ⚠ BREAKING CHANGES

* The `simzoo` package is no longer included and has to be installed separately through the [stable-gym](https://github.com/rickstaa/stable-gym) repository.

### Features

* add hyperparameter variant support for exp config files ([6ea47f0](https://github.com/rickstaa/stable-learning-control/commit/6ea47f0c330e2ea57c98e50165c160ffa43ff53e))
* add small policy test script ([c701d69](https://github.com/rickstaa/stable-learning-control/commit/c701d69c507d735380cbb619037fca4f9a43e8c2))
* add support for dictionary type observation spaces ([e3bf761](https://github.com/rickstaa/stable-learning-control/commit/e3bf76194875576ce6b623a97987696eb7105c0c))


### Bug Fixes

* fix several env/policy load bugs ([dddd4d8](https://github.com/rickstaa/stable-learning-control/commit/dddd4d8343301e322c82285b777a259bb6221cb0))
* fix several policy loading problems ([51a664e](https://github.com/rickstaa/stable-learning-control/commit/51a664e3d18a3b1c315d943f328be66ed5ab683a))
* fixes npm release action ([e47ef0f](https://github.com/rickstaa/stable-learning-control/commit/e47ef0fc3936496f5dfed0515b776812244dfcd6))
* **package.json:** this commit makes sure the release run script contains no bugs ([ef958ca](https://github.com/rickstaa/stable-learning-control/commit/ef958cac744bc47202c39f4addef25d9d142c1cd))
* remove 'simzoo' submodule ([#246](https://github.com/rickstaa/stable-learning-control/issues/246)) ([0122aae](https://github.com/rickstaa/stable-learning-control/commit/0122aaeb79d6c62da48591199e3a4026043c628b))

### [1.3.14](https://github.com/rickstaa/stable-learning-control/compare/v1.3.73...v1.3.14) (2023-06-05)

### [1.3.3](https://github.com/rickstaa/stable-learning-control/compare/v1.3.2...v1.3.3) (2022-02-24)

### [1.3.1](https://github.com/rickstaa/stable-learning-control/compare/v1.3.0...v1.3.1) (2022-02-17)


### Bug Fixes

* fix several policy loading problems ([51a664e](https://github.com/rickstaa/stable-learning-control/commit/51a664e3d18a3b1c315d943f328be66ed5ab683a))

## [1.3.0](https://github.com/rickstaa/stable-learning-control/compare/v1.2.8...v1.3.0) (2022-02-17)


### Features

* add small policy test script ([c701d69](https://github.com/rickstaa/stable-learning-control/commit/c701d69c507d735380cbb619037fca4f9a43e8c2))


### Bug Fixes

* fix several env/policy load bugs ([dddd4d8](https://github.com/rickstaa/stable-learning-control/commit/dddd4d8343301e322c82285b777a259bb6221cb0))

### [1.2.8](https://github.com/rickstaa/stable-learning-control/compare/v1.2.7...v1.2.8) (2022-02-16)

### [1.2.6](https://github.com/rickstaa/stable-learning-control/compare/v1.2.5...v1.2.6) (2022-02-16)

### [1.2.4](https://github.com/rickstaa/stable-learning-control/compare/v1.2.3...v1.2.4) (2022-02-07)

### [1.2.3](https://github.com/rickstaa/stable-learning-control/compare/v1.2.2...v1.2.3) (2022-02-07)

### [1.2.2](https://github.com/rickstaa/stable-learning-control/compare/v1.2.1...v1.2.2) (2022-02-07)

### [1.2.1](https://github.com/rickstaa/stable-learning-control/compare/v1.2.0...v1.2.1) (2022-02-07)

## [1.2.0](https://github.com/rickstaa/stable-learning-control/compare/v1.1.55...v1.2.0) (2022-02-04)


### Features

* add hyperparameter variant support for exp config files ([6ea47f0](https://github.com/rickstaa/stable-learning-control/commit/6ea47f0c330e2ea57c98e50165c160ffa43ff53e))

### [1.1.55](https://github.com/rickstaa/stable-learning-control/compare/v1.1.54...v1.1.55) (2022-02-04)

### [1.1.54](https://github.com/rickstaa/stable-learning-control/compare/v1.1.53...v1.1.54) (2022-02-01)

### [1.1.47](https://github.com/rickstaa/stable-learning-control/compare/v1.1.46...v1.1.47) (2021-12-16)

### [1.1.45](https://github.com/rickstaa/stable-learning-control/compare/v1.1.44...v1.1.45) (2021-12-13)

### [1.1.43](https://github.com/rickstaa/stable-learning-control/compare/v1.1.37...v1.1.43) (2021-12-13)

### [1.1.28](https://github.com/rickstaa/stable-learning-control/compare/v1.1.27...v1.1.28) (2021-10-07)

### [1.1.23](https://github.com/rickstaa/stable-learning-control/compare/v1.1.22...v1.1.23) (2021-09-18)

### [1.1.22](https://github.com/rickstaa/stable-learning-control/compare/v1.1.21...v1.1.22) (2021-09-13)

### [1.1.21](https://github.com/rickstaa/stable-learning-control/compare/v1.1.20...v1.1.21) (2021-09-13)

### [1.1.20](https://github.com/rickstaa/stable-learning-control/compare/v1.1.19...v1.1.20) (2021-09-13)

### [1.1.19](https://github.com/rickstaa/stable-learning-control/compare/v1.1.18...v1.1.19) (2021-09-13)

### [1.1.18](https://github.com/rickstaa/stable-learning-control/compare/v1.1.17...v1.1.18) (2021-09-10)

### [1.1.17](https://github.com/rickstaa/stable-learning-control/compare/v1.1.16...v1.1.17) (2021-09-09)

### [1.1.16](https://github.com/rickstaa/stable-learning-control/compare/v1.1.15...v1.1.16) (2021-09-09)

### [1.1.15](https://github.com/rickstaa/stable-learning-control/compare/v1.1.14...v1.1.15) (2021-09-09)

### [1.1.14](https://github.com/rickstaa/stable-learning-control/compare/v1.1.13...v1.1.14) (2021-09-09)

### [1.1.13](https://github.com/rickstaa/stable-learning-control/compare/v1.1.12...v1.1.13) (2021-09-09)

### [1.1.12](https://github.com/rickstaa/stable-learning-control/compare/v1.1.11...v1.1.12) (2021-09-09)

### [1.1.11](https://github.com/rickstaa/stable-learning-control/compare/v1.1.10...v1.1.11) (2021-09-09)

### [1.1.10](https://github.com/rickstaa/stable-learning-control/compare/v1.1.9...v1.1.10) (2021-09-09)

### [1.1.9](https://github.com/rickstaa/stable-learning-control/compare/v1.1.8...v1.1.9) (2021-09-09)

### [1.1.8](https://github.com/rickstaa/stable-learning-control/compare/v1.1.7...v1.1.8) (2021-09-09)

### [1.1.7](https://github.com/rickstaa/stable-learning-control/compare/v1.1.6...v1.1.7) (2021-09-09)

### [1.1.6](https://github.com/rickstaa/stable-learning-control/compare/v1.1.5...v1.1.6) (2021-09-09)

### [1.1.5](https://github.com/rickstaa/stable-learning-control/compare/v1.1.4...v1.1.5) (2021-09-09)

### [1.1.4](https://github.com/rickstaa/stable-learning-control/compare/v1.1.3...v1.1.4) (2021-09-09)


### Bug Fixes

* fixes npm release action ([e47ef0f](https://github.com/rickstaa/stable-learning-control/commit/e47ef0fc3936496f5dfed0515b776812244dfcd6))

### [1.1.3](https://github.com/rickstaa/stable-learning-control/compare/v1.1.2...v1.1.3) (2021-09-09)

### [1.1.2](https://github.com/rickstaa/stable-learning-control/compare/v1.1.0...v1.1.2) (2021-09-09)


### Bug Fixes

* **package.json:** this commit makes sure the release run script contains no bugs ([ef958ca](https://github.com/rickstaa/stable-learning-control/commit/ef958cac744bc47202c39f4addef25d9d142c1cd))

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Generated by [`auto-changelog`](https://github.com/CookPete/auto-changelog).

## [v1.1.1](https://github.com/rickstaa/stable-learning-control/compare/v1.1.0...v1.1.1) - 2021-06-12

### Merged

*   :arrow_up: Update dependency auto-changelog to v2.3.0 [`#127`](https://github.com/rickstaa/stable-learning-control/pull/127)

### Commits

*   :recycle: Simplifies codebase [`27d0db8`](https://github.com/rickstaa/stable-learning-control/commit/27d0db8ce16d6613f5d061cbebe0cba07462898e)
*   :art: Adds ability to disable baseline in eval_robustness [`47eb9af`](https://github.com/rickstaa/stable-learning-control/commit/47eb9af279b7aab6781e30b23a147118ec5cb360)
*   :bug: Fixes eval_robustness action space clipping problem [`252d7e5`](https://github.com/rickstaa/stable-learning-control/commit/252d7e5694d7474b51622265bce1bbe5551906fb)
*   :wrench: Updates dependencies and doc [`8ffa4fb`](https://github.com/rickstaa/stable-learning-control/commit/8ffa4fb7e3577a9d99a548c99e8c15b1b37e7033)
*   :bug: Fixes pickle bug [`7e87d46`](https://github.com/rickstaa/stable-learning-control/commit/7e87d46cc36793b377dbfe733000e42215386107)
*   :memo: Updates changelog [`b7ab60d`](https://github.com/rickstaa/stable-learning-control/commit/b7ab60db856bf7609ea08f979883024cd20ce783)
*   :green_heart: Updates gh-action cache [`031df8e`](https://github.com/rickstaa/stable-learning-control/commit/031df8efab68c73b578c57a187b67b1bb5e00ae8)
*   :art: Cleansup code base [`8b8920b`](https://github.com/rickstaa/stable-learning-control/commit/8b8920b270a6c40e0d60726f7b734dd7c7ae0046)
*   :bookmark: Bump version: 1.1.0 → 1.1.1 [`aa8abba`](https://github.com/rickstaa/stable-learning-control/commit/aa8abba055f54d4a7dd8ffe1943330c6242ed6ef)
*   :art: Applies some small code format improvements [`e0ed178`](https://github.com/rickstaa/stable-learning-control/commit/e0ed1780b77fced0d762bcb61de851a6be85e633)
*   :alien: Fixes namespace package namespace shorting problem [`af3d8b0`](https://github.com/rickstaa/stable-learning-control/commit/af3d8b0006f3d35e64eebc772ad9abdd952d6495)
*   :art: Fixes some small syntax errors [`8c23933`](https://github.com/rickstaa/stable-learning-control/commit/8c23933e149a88f3bce7203217bccd89a786f2a5)
*   :alien: Fixes namespace package namespace shorting problem [`0edda60`](https://github.com/rickstaa/stable-learning-control/commit/0edda6090900edb7c3e45b3e545d45d9b5568028)
*   :alien: Updates simzoo submodule version [`99bc4e8`](https://github.com/rickstaa/stable-learning-control/commit/99bc4e8c77940bcb768e80311b8b68306e26fc7e)
*   :alien: Updates simzoo submodule [`fb486e8`](https://github.com/rickstaa/stable-learning-control/commit/fb486e884b389f3e040e8ea6d3bb37f66040a664)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`103f011`](https://github.com/rickstaa/stable-learning-control/commit/103f011713f459112af13830c81d24a16b172c91)
*   :alien: Adds a recurring impulse to the disturbances [`9fc15cb`](https://github.com/rickstaa/stable-learning-control/commit/9fc15cbf5ae6165f5c6b840b49cbe1d049fc322c)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`67d9125`](https://github.com/rickstaa/stable-learning-control/commit/67d91259a4aa32a7b13574476830eaf532ebd586)
*   :alien: Updates simzoo submodule [`005c6ed`](https://github.com/rickstaa/stable-learning-control/commit/005c6ed90afc50bd2c559b276f66a4d1bd8a8310)
*   :bulb: Fixes several code comments [`5498802`](https://github.com/rickstaa/stable-learning-control/commit/549880245aaec89afcc6aeb9f0299a189a15b486)
*   :alien: Updates simzoo submodule [`be72356`](https://github.com/rickstaa/stable-learning-control/commit/be72356390f8461926542af2d79e4541c764c508)
*   :alien: Updates simzoo submodule [`e387447`](https://github.com/rickstaa/stable-learning-control/commit/e387447122627bc4210b3c98ccbd3acf574620f7)
*   :alien: Updates simzoo submodule [`615c57b`](https://github.com/rickstaa/stable-learning-control/commit/615c57b0d01c36378a6e89bc35857e25a0e8f596)
*   :alien: Updates simzoo submodule [`573a14b`](https://github.com/rickstaa/stable-learning-control/commit/573a14bfbbcdbb3df1b78be5bcb2c8eead4aa271)
*   :alien: Updates simzoo submodule [`fbb93f5`](https://github.com/rickstaa/stable-learning-control/commit/fbb93f5a833034746784ceb74c5017853177f73d)

## [v1.1.0](https://github.com/rickstaa/stable-learning-control/compare/v1.0.3...v1.1.0) - 2021-04-30

### Commits

*   :sparkles: Adds script version of the robustness eval tool [`bef8595`](https://github.com/rickstaa/stable-learning-control/commit/bef8595ee7096721c0b1dbb2b53486cba7291523)
*   :sparkles: Adds a trajectory buffer [`6295ff2`](https://github.com/rickstaa/stable-learning-control/commit/6295ff224c0eb87849c644c53cf27078066a17ee)
*   :sparkles: Adds adv, ret and logp to TrajectoryBuffer [`34dc5e2`](https://github.com/rickstaa/stable-learning-control/commit/34dc5e2fbab071dad2615c0708a7d5321cfab52c)
*   :memo: Updates Changelog [`83c6b5f`](https://github.com/rickstaa/stable-learning-control/commit/83c6b5f160315d9867ee5e284e1d471a1beadbac)
*   :art: Cleans up Trajectory buffer [`72419a3`](https://github.com/rickstaa/stable-learning-control/commit/72419a3218b4cf29026b4b4a6232879c0b9d8674)
*   :sparkles: Adds pytorch trajectory buffer [`1cde5b0`](https://github.com/rickstaa/stable-learning-control/commit/1cde5b02d8ca5144eded36a79bcc6d793c6486bc)
*   :sparkles: Adds non-homogenious disturbance [`8c2e2a7`](https://github.com/rickstaa/stable-learning-control/commit/8c2e2a76bd2b13a3523ada59616cb5c026d48e05)
*   :art: Clean-up code [`8f41432`](https://github.com/rickstaa/stable-learning-control/commit/8f41432f588e8c7ffdec0e07898d44ad5907ead2)
*   :bookmark: Bump version: 1.0.3 → 1.1.0 [`81468b3`](https://github.com/rickstaa/stable-learning-control/commit/81468b3efeee0c4af2526e94ed7dfb19f998c0af)
*   :alien: Updates simzoo submodule [`238cb72`](https://github.com/rickstaa/stable-learning-control/commit/238cb72484d60120f8808efff2872471d26507f9)
*   :alien: Updates simzoo submodule [`8063a4f`](https://github.com/rickstaa/stable-learning-control/commit/8063a4f1e3265e4b3543eeb3ec56c46655274a68)
*   :alien: Updates simzoo submodule [`dd0fdeb`](https://github.com/rickstaa/stable-learning-control/commit/dd0fdeb8a4b0d3ade05abccb92bb6e41d5e30eaf)

## [v1.0.3](https://github.com/rickstaa/stable-learning-control/compare/v1.0.2...v1.0.3) - 2021-04-21

### Merged

*   :sparkles: Add combined disturbance [`#125`](https://github.com/rickstaa/stable-learning-control/pull/125)

### Commits

*   :rewind: Removes gpl algorithms from master branch [`e80b578`](https://github.com/rickstaa/stable-learning-control/commit/e80b5780f22ce6eafc74f8c04a268959b9e83208)
*   :construction: Adds 6th lac variant [`af194d4`](https://github.com/rickstaa/stable-learning-control/commit/af194d4202c0de651853b2b6bd98d24a5b6e711e)
*   :bug: Applies some small bug fixes [`0712e20`](https://github.com/rickstaa/stable-learning-control/commit/0712e2049c451d6c207d4dbdef9f8be1e6302974)
*   :memo: Updates documentation [`14b7a53`](https://github.com/rickstaa/stable-learning-control/commit/14b7a5305fa56f2ef911ba8660e1249d0ff8e768)
*   :memo; Updates documentation [`213c552`](https://github.com/rickstaa/stable-learning-control/commit/213c5524171f4bd396ed7eecb0c7d3bb9028d695)
*   :fire: Removes redundant lac algorithm [`c0d82a2`](https://github.com/rickstaa/stable-learning-control/commit/c0d82a26248fae28226765366352b30a4286b238)
*   :sparkles: Adds disturbance type/variant list option [`1e1c1aa`](https://github.com/rickstaa/stable-learning-control/commit/1e1c1aab3719ba694cb2a4a1952fdf2b91cc9a08)
*   :sparkles: Adds merged option to robustness_eval utility [`7d585cc`](https://github.com/rickstaa/stable-learning-control/commit/7d585cc947899cd6e673a47a12cc66d8a09ca6a2)
*   :sparkles: Adds disturber to CartPoleCost env [`63fe0d7`](https://github.com/rickstaa/stable-learning-control/commit/63fe0d7374775dcd397328c10c19c98c40cbd971)
*   :twisted_rightwards_arrows: :memo: Removes sphinx warnings/errors [`cf8ba4c`](https://github.com/rickstaa/stable-learning-control/commit/cf8ba4c462e33f98624ea92d7046ee01a428f33f)
*   :memo: Removes sphinx warnings/errors [`dd45fa3`](https://github.com/rickstaa/stable-learning-control/commit/dd45fa3ed6a96fdb411a3a2ea98c114abf72d028)
*   :bug: Fixes bugs inside the test_policy and eval_robustness utilities [`c7c4daf`](https://github.com/rickstaa/stable-learning-control/commit/c7c4dafeaaadc9bc7d990cf6b02b200db0a42c46)
*   :construction: Adds lac5 to cmd line [`e2f06eb`](https://github.com/rickstaa/stable-learning-control/commit/e2f06eb1f5a26e595a52805889f0021766bbfba1)
*   :art: :wrench: Fix experiment files [`f5447f3`](https://github.com/rickstaa/stable-learning-control/commit/f5447f35dd637cb9187efd0292d5fe2b56075ec2)
*   :bug: Fixes some small bugs [`298148d`](https://github.com/rickstaa/stable-learning-control/commit/298148d24bbe4fe3fee09082861d625145da2d76)
*   :zap: Removes redundant grad freeze operation [`4d8e726`](https://github.com/rickstaa/stable-learning-control/commit/4d8e726c538925e7cd6341b9d000f14fe06f9875)
*   :construction: Adds Lyapunov constraint to the Lyapunov critic [`66ff754`](https://github.com/rickstaa/stable-learning-control/commit/66ff754885ea8d2315bf77e62fb6d7578aaa47af)
*   :twisted_rightwards_arrows: :memo: Updates eval robustness tool naming [`40b3adf`](https://github.com/rickstaa/stable-learning-control/commit/40b3adf8c2d93a29c609b06ad56f344ed2123ec1)
*   :bug: Fixes eval_robustness max_episode_steps problem [`93b8536`](https://github.com/rickstaa/stable-learning-control/commit/93b8536df241b08d57210d5b448af3249ca9b9b4)
*   :fire: Removes redundant experiment file [`7fc1fd4`](https://github.com/rickstaa/stable-learning-control/commit/7fc1fd4403c6dad42f5322fbe6b108b72ba031dd)
*   :sparkles: Adds disturbance type/variant list option [`2a2f960`](https://github.com/rickstaa/stable-learning-control/commit/2a2f960d0c17b473ed0ea9c42ae2a07e17fff66f)
*   :twisted_rightwards_arrows: :bug: Fixes some small eval robustness bugs [`00b47ca`](https://github.com/rickstaa/stable-learning-control/commit/00b47caeadd6f26f45f5c819b8a3f132285d88d9)
*   :loud_sound: Update std_out log variables [`72843e8`](https://github.com/rickstaa/stable-learning-control/commit/72843e8b0cbc8d6fc227190a90a8a9167de8d79f)
*   :fire: Removes redundant file [`d7c7e15`](https://github.com/rickstaa/stable-learning-control/commit/d7c7e15b42c0304348e9aefe013e517df6d304bf)
*   :bookmark: Bump version: 1.0.2 → 1.0.3 [`893244d`](https://github.com/rickstaa/stable-learning-control/commit/893244d78eb19364b442fdcac4bc7b51a5031ab4)
*   :building_construction: Improves package architecture [`62cde45`](https://github.com/rickstaa/stable-learning-control/commit/62cde45b3777dba680992a4baa3d3a5bcc44f698)
*   :art: Makes some small code changes [`1b1cc77`](https://github.com/rickstaa/stable-learning-control/commit/1b1cc779b963cf3695215cd5eb45fbd5322d2acd)
*   :bug: Fixes eval_robustness plot bug [`b53fb76`](https://github.com/rickstaa/stable-learning-control/commit/b53fb760924a515650fd3184f4935695ef78ad30)
*   :fire: Cleans up redundant code [`78583e2`](https://github.com/rickstaa/stable-learning-control/commit/78583e2797a6b0abb7b5efba0bb925a92865ac1b)
*   :bug: Fixes lac4 critic loss function [`f10b6dc`](https://github.com/rickstaa/stable-learning-control/commit/f10b6dc193014e61a185d285d1d1c27eab47ce41)
*   :bug: Fixes eval_robustness observation plot bug [`7df0aaf`](https://github.com/rickstaa/stable-learning-control/commit/7df0aafc6805d7403c8beb1992a09f31604604ac)
*   :alien: Updates changelog [`768f6f2`](https://github.com/rickstaa/stable-learning-control/commit/768f6f2c58d8f1a5103417aeb1b87b208672bbf8)
*   :alien: Updates simzoo submodule [`8b3d654`](https://github.com/rickstaa/stable-learning-control/commit/8b3d654874219102b36c4501df4e6e2efe658063)
*   :pencil2: Fixes small typos [`e950bfb`](https://github.com/rickstaa/stable-learning-control/commit/e950bfb8f78081fa99c6d496605ad99fcb2062f7)
*   :alien: Updates simzoo submodule [`96c1822`](https://github.com/rickstaa/stable-learning-control/commit/96c18221b856242cc4035ca950ad6f1fad25cb44)
*   :alien: Updates simzoo [`604e1f2`](https://github.com/rickstaa/stable-learning-control/commit/604e1f25cdaeeb217a135eba9db99d8dc80c58b0)
*   :alien: Updates simzoo submodule [`afa67f2`](https://github.com/rickstaa/stable-learning-control/commit/afa67f2954119b9ce810c28a11be1c262238365b)
*   :alien: Updates simzoo submodule [`f2cd1ae`](https://github.com/rickstaa/stable-learning-control/commit/f2cd1ae77d910d6342ce8ed7d03593e2bb7d9bc2)
*   :twisted_rightwards_arrows: Merge branch 'gpl' into main [`9c0bd8e`](https://github.com/rickstaa/stable-learning-control/commit/9c0bd8ea307d100702bb28a827d9c7128737725f)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`80abf92`](https://github.com/rickstaa/stable-learning-control/commit/80abf92c396564ee5e1f6d19475a0bcb0fd6fe39)
*   :bug: Fixes lac6 import bug [`cbb2bdb`](https://github.com/rickstaa/stable-learning-control/commit/cbb2bdb1d2f3d7f068afdbd7e4c11eec5c509254)
*   :wrench: Updates setup.cfg file [`aafdb91`](https://github.com/rickstaa/stable-learning-control/commit/aafdb91d33329a841486ab2fae36950fa6fa9c30)
*   :alien: Updates simzoo submodule [`a87d4f4`](https://github.com/rickstaa/stable-learning-control/commit/a87d4f4ebb140f1637cbe0e74b50503fc651d2a2)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`761d9f8`](https://github.com/rickstaa/stable-learning-control/commit/761d9f8b65b79b26b9229ea08ccb691dcee09f48)
*   :memo: Updates documentation [`715b438`](https://github.com/rickstaa/stable-learning-control/commit/715b438fc8666df20b0c238950dceeb92caf02d0)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`b80f648`](https://github.com/rickstaa/stable-learning-control/commit/b80f648e9ee59e22e8e7360e197c6203f4978e91)
*   :memo: Updates documentation [`415ac8d`](https://github.com/rickstaa/stable-learning-control/commit/415ac8da05f6c7e28c358ee0351c19a8e9171961)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`e3fa140`](https://github.com/rickstaa/stable-learning-control/commit/e3fa140d4c7c0e36602564b36665e681831fa714)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`70f0bac`](https://github.com/rickstaa/stable-learning-control/commit/70f0bac472341a2c66fba6726178331b766e60a3)
*   :alien: Updates simzoo [`51524e1`](https://github.com/rickstaa/stable-learning-control/commit/51524e159b234fff02cc8011c4da56463a420ccc)
*   :art: :alien: Updates simzoo [`44fd0bf`](https://github.com/rickstaa/stable-learning-control/commit/44fd0bf599873bf1e26dc35b7ecb2c17eb692ba7)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`90e5c26`](https://github.com/rickstaa/stable-learning-control/commit/90e5c269fcfd6d322e9f9e8f6b910c140f519250)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`290cfe9`](https://github.com/rickstaa/stable-learning-control/commit/290cfe99088f8f471aaebb8e3b6e2214c7ba6ddb)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`bf8009f`](https://github.com/rickstaa/stable-learning-control/commit/bf8009f2bb9f891220faf01ed0b91a7703c02c0c)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`c2c2231`](https://github.com/rickstaa/stable-learning-control/commit/c2c22311483e50ca1a69b0e37fd15730839dea00)
*   :construction: Adds new lac variant [`52051df`](https://github.com/rickstaa/stable-learning-control/commit/52051dfbf9812dd37fa290187d07bd716b1f763f)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`4f595f1`](https://github.com/rickstaa/stable-learning-control/commit/4f595f118f84222aeb7beec573393a77c5f88719)
*   :bug: Fixes environment ep_len bug [`334ab91`](https://github.com/rickstaa/stable-learning-control/commit/334ab91d79943d768e692cdcd50a23bd81268530)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`561a2de`](https://github.com/rickstaa/stable-learning-control/commit/561a2ded7011412c7ab9baffe0cff95965b82490)
*   :construction: Adds new LAC and SAC versions [`24974e7`](https://github.com/rickstaa/stable-learning-control/commit/24974e7898c6446a0b47d0fbe3d130e594aa6a18)
*   :sparkles: Adds initial gpl changes [`c80ccd2`](https://github.com/rickstaa/stable-learning-control/commit/c80ccd26f59a9acc7d2eaa6d700db5710b263821)
*   :memo: Adds small hello document [`1618687`](https://github.com/rickstaa/stable-learning-control/commit/1618687033324a68e83665584b0de1bbb607fdb1)
*   :bug: Fixes CartPoleCost cost function [`b6afb5c`](https://github.com/rickstaa/stable-learning-control/commit/b6afb5c1e410759bb7fabaff0eec08b6f69d2b4b)
*   :construction: Adds initial version of the improved lac algorithm [`c2b0f2a`](https://github.com/rickstaa/stable-learning-control/commit/c2b0f2afda6eba91ae9b4d209f624b901c72fd04)
*   :twisted_rightwards_arrows: :bug: Fixes network architecture load bug in the test_policy utility [`7c2282b`](https://github.com/rickstaa/stable-learning-control/commit/7c2282b8f9ea5eaac6c247f90e6290feefcad860)
*   :bug: Fixes network architecture load bug in the test_policy utility [`56de198`](https://github.com/rickstaa/stable-learning-control/commit/56de198b7d7db99bc7100b5f7fd562863d79b02e)
*   :sparkles: Adds lac and sac changes to lac2 and sac2 [`83d9488`](https://github.com/rickstaa/stable-learning-control/commit/83d9488331c571bedb4ad389aa9dc0213fafb882)
*   :construction: Adds several new LAC versions [`cbd2582`](https://github.com/rickstaa/stable-learning-control/commit/cbd25824cb0b2dee8128713967937cda3e90dcbc)
*   :bug: Fixes environment max_ep_len problem [`3c17635`](https://github.com/rickstaa/stable-learning-control/commit/3c17635684c7a8248f52840b1e9ab864811ad70c)
*   :art: :wrench: Adds Han experiment files [`19fface`](https://github.com/rickstaa/stable-learning-control/commit/19fface45d3a51c10791cbc9f3553cd63e6a6e7d)
*   :bug: Fixes max_ep_len bug [`7e25b36`](https://github.com/rickstaa/stable-learning-control/commit/7e25b3640f484d7b50d1b35dc8aa59573f278918)
*   :wrench: Adds Han et al config files [`80b0c54`](https://github.com/rickstaa/stable-learning-control/commit/80b0c549c09c9215431379993c55206336f00d1f)
*   :wrench: Updates gamma to the right value [`68d9561`](https://github.com/rickstaa/stable-learning-control/commit/68d95619da879f0e323766cd89767dd6d0bf9de4)
*   :art: Improves variable naming in LAC and SAC algorithms [`005cbd8`](https://github.com/rickstaa/stable-learning-control/commit/005cbd86dbc6d9e9de13ce9dd6ce08e12990ce37)
*   :bug: Fixes SAC minimization environment not training problem [`3cb70f1`](https://github.com/rickstaa/stable-learning-control/commit/3cb70f1d9e6cb138c911177994882d63ad04b57b)
*   :bug: Fixes SAC minimization environment not training problem [`1d7ec07`](https://github.com/rickstaa/stable-learning-control/commit/1d7ec07d805e6452083cd41a62a5ac382e8c67c2)
*   :bulb: Updates algorithms code comments [`65c5f70`](https://github.com/rickstaa/stable-learning-control/commit/65c5f7073be6d66dc7940da9e0924a0c442603f3)
*   :fire: Cleans up unused code [`10810ea`](https://github.com/rickstaa/stable-learning-control/commit/10810ea468abbfe8d8855c615384651a4c34c870)
*   :bug: Fixes logger dump_tabular bug [`bd1d536`](https://github.com/rickstaa/stable-learning-control/commit/bd1d536fd8137ef864ac60dbe0f70904b4c64bad)
*   :art: Improves variable naming in the LAC algorithm [`e788050`](https://github.com/rickstaa/stable-learning-control/commit/e78805093f31c4ec09f6841d6f15dcb4fac9364c)
*   :memo: Adds CHANGELOG [`4694d25`](https://github.com/rickstaa/stable-learning-control/commit/4694d25cc67182f42dd2faa6d30e2be97ede9aeb)
*   :bulb: Updates algorithm code comments [`d40a6ac`](https://github.com/rickstaa/stable-learning-control/commit/d40a6ac875547b84cc25c634f4df38e585777c2c)
*   :bulb: Updates SAC code comments [`8d4c4cd`](https://github.com/rickstaa/stable-learning-control/commit/8d4c4cdc0b6628c9d7370b4f627ffb163ad8e8d6)
*   :wrench: Updates Han exp config files [`9c1f341`](https://github.com/rickstaa/stable-learning-control/commit/9c1f3413f31282b841f079a3b4dbd26c52c47142)
*   :wrench: Updates Han exp config files [`3c00b26`](https://github.com/rickstaa/stable-learning-control/commit/3c00b265099d6b442f39390695e4b2ee26fa79c1)
*   :mute: Removes simzoo logging statements [`397debd`](https://github.com/rickstaa/stable-learning-control/commit/397debdbdc18bab99546cc975861e5def40f2f8e)
*   :pencil2: Fixes lac2 name [`f73de6b`](https://github.com/rickstaa/stable-learning-control/commit/f73de6b7b309a0ff8130d1dd9abdea5f0cb790ba)
*   :art: Removes unnecessary pytorch detach() calls [`2777860`](https://github.com/rickstaa/stable-learning-control/commit/27778609b5a3b96566f6b8a850c8d81a7d1fb6d2)
*   :wrench: Update Han exp config files [`b3b5e36`](https://github.com/rickstaa/stable-learning-control/commit/b3b5e36cc6ab2bc2049896538466313103d501e4)
*   :fire: Removes redundant code [`f1a65e6`](https://github.com/rickstaa/stable-learning-control/commit/f1a65e6de6e7e95055a0093f0b54af7895dcb02a)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`139fb07`](https://github.com/rickstaa/stable-learning-control/commit/139fb07de88eaec34d9b1bf09448ec69bb40252b)
*   :bug: Removes bug from Han et al sac cfg file [`64e82ee`](https://github.com/rickstaa/stable-learning-control/commit/64e82eef5d545ee8aa4733c25c9036694283f5d6)
*   :art: Improves LAC code structure [`2a5357f`](https://github.com/rickstaa/stable-learning-control/commit/2a5357f4f963a4b89835b943b791af3a2066c5d4)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`78fec47`](https://github.com/rickstaa/stable-learning-control/commit/78fec47c17d4db8b500d233cbfe6341929786b02)
*   :bulb: Updates SAC code comments [`8ce3549`](https://github.com/rickstaa/stable-learning-control/commit/8ce354983778dc5841d6e88650ccc66a753511a2)
*   :bug: Removes bug from Han et al sac cfg file [`88da027`](https://github.com/rickstaa/stable-learning-control/commit/88da027c45a478d6c081f9c671c7f8ae7859c77c)
*   :wrench: Update Han exp config files [`8815229`](https://github.com/rickstaa/stable-learning-control/commit/88152296f409b21341e3fc4c3d807bd4079d94ea)
*   :bulb: Updates code comments [`b3c38d8`](https://github.com/rickstaa/stable-learning-control/commit/b3c38d815f07b60f55fab219658efb271f9aa68e)
*   :alien: Updates simzoo submodule [`455ca60`](https://github.com/rickstaa/stable-learning-control/commit/455ca60402fe5f0d6f742227e13da17e587ef011)
*   :alien: Updates simzoo submodule [`48b6e31`](https://github.com/rickstaa/stable-learning-control/commit/48b6e313db7944dcf557b4e012fb07c4adf4b91c)
*   :bug: Fixes simzoo gym environments registration bug [`59e716e`](https://github.com/rickstaa/stable-learning-control/commit/59e716eacd2b0debc0cdf3c55ae24cbd123b1259)
*   :bug: Fixes simzoo gym environments registration bug [`02c88f1`](https://github.com/rickstaa/stable-learning-control/commit/02c88f1261a4d75f8efcc352797add79d2ba68cb)
*   :fire: Removes old CartPole environment [`3ca2ea3`](https://github.com/rickstaa/stable-learning-control/commit/3ca2ea37cdd90b60495a5f102164977f2c7671fa)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`7f0d158`](https://github.com/rickstaa/stable-learning-control/commit/7f0d15837ea2f88aa9af5a6bc26198c684e21dd7)
*   :loud_sound: Adds environment information logging [`df1fc2c`](https://github.com/rickstaa/stable-learning-control/commit/df1fc2c02fa433e6c20a84ecf4ef159f6e693bcb)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`958c783`](https://github.com/rickstaa/stable-learning-control/commit/958c783398a1ead535cf7a4bf584a846030e127e)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`f6b1857`](https://github.com/rickstaa/stable-learning-control/commit/f6b18574a9fdaeb70c421865c81dd1c0688ab510)
*   :alien: Updates simzoo submodule [`5f25fb3`](https://github.com/rickstaa/stable-learning-control/commit/5f25fb33dc4ce4183666a0d34523ecefede1440f)
*   :bug: Fixes the CartPoleCost environment cost function [`95f728e`](https://github.com/rickstaa/stable-learning-control/commit/95f728ea9c563e267fb96974b0a5c3b918a74e90)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`853e7c7`](https://github.com/rickstaa/stable-learning-control/commit/853e7c7f6e39789e4836c7f4f16fb5aeaa7f29cb)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`dc8aee3`](https://github.com/rickstaa/stable-learning-control/commit/dc8aee333cf74e4a8bb048f8d45406b5cc4b8316)
*   :twisted_rightwards_arrows: Merge branch 'main' into gpl [`ddb865e`](https://github.com/rickstaa/stable-learning-control/commit/ddb865e629dd5ddc20b29d7d4b7834252c7ead52)
*   Write hello back [`a16789a`](https://github.com/rickstaa/stable-learning-control/commit/a16789a9658a3660502904dc3cf17f6e6ba1b5de)

## [v1.0.2](https://github.com/rickstaa/stable-learning-control/compare/v1.0.1...v1.0.2) - 2021-03-31

### Commits

*   :memo: Updates Simzoo environments documentation [`79e92f7`](https://github.com/rickstaa/stable-learning-control/commit/79e92f72fb4d51178a43bd297bf2d2b9074a187b)
*   :alien: Updates the Simzoo submodule and documentation [`9c23bb1`](https://github.com/rickstaa/stable-learning-control/commit/9c23bb19046b0b4c5e711c43edd97516858c6785)
*   :alien: Updates the Simzoo submodule [`c791279`](https://github.com/rickstaa/stable-learning-control/commit/c7912790e7d9c8f3d36cc73179c0bf3e1ca72e96)
*   :memo: Updates changelog [`1c989c7`](https://github.com/rickstaa/stable-learning-control/commit/1c989c71ac0f84a093e76af74656f5db5e1f7721)
*   :bookmark: Bump version: 1.0.1 → 1.0.2 [`3f91fa1`](https://github.com/rickstaa/stable-learning-control/commit/3f91fa109ecf2baa70e9fe3a09aa9d8bbb3d2564)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`f2fc225`](https://github.com/rickstaa/stable-learning-control/commit/f2fc225b3965be8fd1b8e195e750f50d56062854)
*   :art: Updates code formatting [`f79c0dc`](https://github.com/rickstaa/stable-learning-control/commit/f79c0dcf1d34a181a1660b8e9d0a418e78241b6d)

## [v1.0.1](https://github.com/rickstaa/stable-learning-control/compare/v1...v1.0.1) - 2021-03-30

### Merged

*   :art: Puts SAC algorithm in seperate file [`#121`](https://github.com/rickstaa/stable-learning-control/pull/121)
*   :bug: Fixes eval_robustness bug [`#120`](https://github.com/rickstaa/stable-learning-control/pull/120)

### Commits

*   :sparkles: Adds logger kwargs to the CLI [`db3d4f5`](https://github.com/rickstaa/stable-learning-control/commit/db3d4f58ba0565f3eba015dae021d399cc409eb3)
*   :white_check_mark: Adds gym environment test script [`957a5ea`](https://github.com/rickstaa/stable-learning-control/commit/957a5ea0d92033407f6d423b3d10c4484dc325d8)
*   :sparkles: Makes SLC compatible with 1D action spaces [`9390213`](https://github.com/rickstaa/stable-learning-control/commit/93902135053d2fa145df9407b846991e737dd023)
*   :green_heart: Remove cache operation from release gh-action [`95d10e2`](https://github.com/rickstaa/stable-learning-control/commit/95d10e2d7aa5b1017f591fc109db58aaa1f5545e)
*   :bookmark: Bump version: v1.0.0 → 1.0.1 [`5a054be`](https://github.com/rickstaa/stable-learning-control/commit/5a054bed02b3e124b17c239e339fb802cf03540b)
*   :bug: Fixes syntax error in the logger [`b0ac672`](https://github.com/rickstaa/stable-learning-control/commit/b0ac672d18a6eb8a96ee858d4be06860d067c60e)
*   :alien: Adds Cartpole environment [`34cc65a`](https://github.com/rickstaa/stable-learning-control/commit/34cc65abc95748c29c3aaf24f5ae469b4e211cf5)
*   :memo: Adds Logger flags to the documentation [`e80a178`](https://github.com/rickstaa/stable-learning-control/commit/e80a178abef443e82eef01267e3cdaaa97877da2)
*   :bug: Fixes a small bug in the simzoo package [`a553900`](https://github.com/rickstaa/stable-learning-control/commit/a553900a9b5f92700e2068669b801ee9fc543389)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`12315b3`](https://github.com/rickstaa/stable-learning-control/commit/12315b3c9b91f96e7480bffb0d7ddd4690b91f46)

## [v1](https://github.com/rickstaa/stable-learning-control/compare/v1.0...v1) - 2021-03-25

## [v1.0](https://github.com/rickstaa/stable-learning-control/compare/v1.0.0...v1.0) - 2021-03-25

## [v1.0.0](https://github.com/rickstaa/stable-learning-control/compare/v0.9.6...v1.0.0) - 2021-03-25

### Commits

*   :memo: Updates CHANGELOG.md [`586a5d1`](https://github.com/rickstaa/stable-learning-control/commit/586a5d18b9ab8d26b6f104cc173524e29b7d9380)
*   :bookmark: Updates code version to v1.0.0 [`d97263b`](https://github.com/rickstaa/stable-learning-control/commit/d97263b53816be82633b9e8f12d4453b93445ee0)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`5a8ebc3`](https://github.com/rickstaa/stable-learning-control/commit/5a8ebc3f7819711c8fe83190219d40d7a2ba0e57)
*   :green_heart: Fixed docs release github action [`23187f0`](https://github.com/rickstaa/stable-learning-control/commit/23187f0169aec04b073eb0d17116a0aeb0967296)

## [v0.9.6](https://github.com/rickstaa/stable-learning-control/compare/v0.9.5...v0.9.6) - 2021-03-25

### Commits

*   :memo: Updates CHANGELOG.md [`0debf73`](https://github.com/rickstaa/stable-learning-control/commit/0debf73a8ac192ea7c65ed620f34e78ebc6e1f5e)
*   :bookmark: Updates code version to v0.9.6 [`204b575`](https://github.com/rickstaa/stable-learning-control/commit/204b575d7365e929ab296f79a8264da7ae8d9825)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`f9f75f4`](https://github.com/rickstaa/stable-learning-control/commit/f9f75f48ca65701959182776df4026c37c1d48a1)
*   :green_heart: Trims down docs release gh-action size [`d3ad44f`](https://github.com/rickstaa/stable-learning-control/commit/d3ad44f17934c80120e468bd0fb605ebd0bc4143)

## [v0.9.5](https://github.com/rickstaa/stable-learning-control/compare/v0.9.4...v0.9.5) - 2021-03-25

### Commits

*   :memo: Updates CHANGELOG.md [`9afc8a5`](https://github.com/rickstaa/stable-learning-control/commit/9afc8a518dc81f0500f78c89c48ae039414b4d86)
*   :bookmark: Updates code version to v0.9.5 [`248e0af`](https://github.com/rickstaa/stable-learning-control/commit/248e0af0e0e6ca48d2cb56e071fd31f03acb151a)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`6c2a0a6`](https://github.com/rickstaa/stable-learning-control/commit/6c2a0a6dd2afbb36e8b0efdcc9a664771226ac10)
*   :green_heart: Adds texlive-full to docs build gh-action [`d61ed01`](https://github.com/rickstaa/stable-learning-control/commit/d61ed0150dcbf78b23335d96a09c126754178a2e)

## [v0.9.4](https://github.com/rickstaa/stable-learning-control/compare/v0.9.3...v0.9.4) - 2021-03-25

### Commits

*   :memo: Updates CHANGELOG.md [`2b46dd1`](https://github.com/rickstaa/stable-learning-control/commit/2b46dd1570a5fc823b726711008506aa30d7c772)
*   :green_heart: Add sphinx-latexpdf container to docs release gh-action [`ba7088d`](https://github.com/rickstaa/stable-learning-control/commit/ba7088df6872fff630f872c615584327d3ae1281)
*   :bookmark: Updates code version to v0.9.4 [`6c1e72a`](https://github.com/rickstaa/stable-learning-control/commit/6c1e72ad70ecca4b6f569fb8aca536a2d8415f96)
*   :wrench: Updates docs make file [`31217d4`](https://github.com/rickstaa/stable-learning-control/commit/31217d4077f5d27e9c42e24a791eef25c4a9ce06)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`99ff0d5`](https://github.com/rickstaa/stable-learning-control/commit/99ff0d5b6e5bca3b635604094bf823ff56be5e5e)
*   :bulb: Updates code comments [`67ae469`](https://github.com/rickstaa/stable-learning-control/commit/67ae469e409b7a18b614973128c8682f4c78f564)

## [v0.9.3](https://github.com/rickstaa/stable-learning-control/compare/v0.9.2...v0.9.3) - 2021-03-25

### Commits

*   :memo: Updates CHANGELOG.md [`90d4894`](https://github.com/rickstaa/stable-learning-control/commit/90d4894508123a16444661789b0eaec27c1845c4)
*   :wrench: Switched from katex latex engine to imgmath [`a337802`](https://github.com/rickstaa/stable-learning-control/commit/a3378026cbc843ba0043a46625cabfc4e7d45d76)
*   :bookmark: Updates code version to v0.9.3 [`91f5600`](https://github.com/rickstaa/stable-learning-control/commit/91f560080eda0cc1e909b2542d983e8611552739)

## [v0.9.2](https://github.com/rickstaa/stable-learning-control/compare/v0.9.1...v0.9.2) - 2021-03-25

### Merged

*   :bug: Fixes --exp_cfg run bug [`#118`](https://github.com/rickstaa/stable-learning-control/pull/118)

### Commits

*   :memo: Updates CHANGELOG.md [`5c442f2`](https://github.com/rickstaa/stable-learning-control/commit/5c442f2cf3dceaa78dca319484c41da6f88772ea)
*   :green_heart: Updates gh-actions so that they also run on main push [`d3279bf`](https://github.com/rickstaa/stable-learning-control/commit/d3279bfcf36de8b87fd14d33be0f399f5bc858a5)
*   :memo: Adds MPI install instructions to docs [`2f1a210`](https://github.com/rickstaa/stable-learning-control/commit/2f1a210952fd0947863744b734c7ff383dd7682a)
*   :bookmark: Updates code version to v0.9.2 [`38d537a`](https://github.com/rickstaa/stable-learning-control/commit/38d537a1b18d31a900eb4af491c45f0dd0de4ab8)
*   :fire: Removes redundant readme's [`76db499`](https://github.com/rickstaa/stable-learning-control/commit/76db49934dd7b774593a71572880445c5447901d)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`711d62b`](https://github.com/rickstaa/stable-learning-control/commit/711d62b4bf29c3beff140dc5088ef3f51243ca15)

## [v0.9.1](https://github.com/rickstaa/stable-learning-control/compare/v0.9...v0.9.1) - 2021-03-24

## [v0.9](https://github.com/rickstaa/stable-learning-control/compare/v0.9.0...v0.9) - 2021-03-25

### Merged

*   :bug: Fixes --exp_cfg run bug [`#118`](https://github.com/rickstaa/stable-learning-control/pull/118)
*   :green_heart: Makes sure OS actions don't run on doc change [`#117`](https://github.com/rickstaa/stable-learning-control/pull/117)
*   :memo: Updates documentation header [`#116`](https://github.com/rickstaa/stable-learning-control/pull/116)
*   :green_heart: Fixes OS gh actions [`#115`](https://github.com/rickstaa/stable-learning-control/pull/115)

### Commits

*   :memo: Updates CHANGELOG.md [`0debf73`](https://github.com/rickstaa/stable-learning-control/commit/0debf73a8ac192ea7c65ed620f34e78ebc6e1f5e)
*   :bookmark: Updates code version to v0.9.6 [`204b575`](https://github.com/rickstaa/stable-learning-control/commit/204b575d7365e929ab296f79a8264da7ae8d9825)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`f9f75f4`](https://github.com/rickstaa/stable-learning-control/commit/f9f75f48ca65701959182776df4026c37c1d48a1)
*   :memo: Updates CHANGELOG.md [`664159b`](https://github.com/rickstaa/stable-learning-control/commit/664159bba6192de000c7c1fe70b4ccd8e40e1ce9)
*   :memo: Updates CHANGELOG.md [`5c442f2`](https://github.com/rickstaa/stable-learning-control/commit/5c442f2cf3dceaa78dca319484c41da6f88772ea)
*   :memo: Updates CHANGELOG.md [`9afc8a5`](https://github.com/rickstaa/stable-learning-control/commit/9afc8a518dc81f0500f78c89c48ae039414b4d86)
*   :memo: Updates CHANGELOG.md [`2b46dd1`](https://github.com/rickstaa/stable-learning-control/commit/2b46dd1570a5fc823b726711008506aa30d7c772)
*   :memo: Updates CHANGELOG.md [`90d4894`](https://github.com/rickstaa/stable-learning-control/commit/90d4894508123a16444661789b0eaec27c1845c4)
*   :wrench: Switched from katex latex engine to imgmath [`a337802`](https://github.com/rickstaa/stable-learning-control/commit/a3378026cbc843ba0043a46625cabfc4e7d45d76)
*   :green_heart: Updates gh-actions so that they also run on main push [`d3279bf`](https://github.com/rickstaa/stable-learning-control/commit/d3279bfcf36de8b87fd14d33be0f399f5bc858a5)
*   :green_heart: Trims down docs release gh-action size [`d3ad44f`](https://github.com/rickstaa/stable-learning-control/commit/d3ad44f17934c80120e468bd0fb605ebd0bc4143)
*   :memo: Adds MPI install instructions to docs [`2f1a210`](https://github.com/rickstaa/stable-learning-control/commit/2f1a210952fd0947863744b734c7ff383dd7682a)
*   :green_heart: Adds texlive-full to docs build gh-action [`d61ed01`](https://github.com/rickstaa/stable-learning-control/commit/d61ed0150dcbf78b23335d96a09c126754178a2e)
*   :bookmark: Updates code version to v0.9.5 [`248e0af`](https://github.com/rickstaa/stable-learning-control/commit/248e0af0e0e6ca48d2cb56e071fd31f03acb151a)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`6c2a0a6`](https://github.com/rickstaa/stable-learning-control/commit/6c2a0a6dd2afbb36e8b0efdcc9a664771226ac10)
*   :green_heart: Add sphinx-latexpdf container to docs release gh-action [`ba7088d`](https://github.com/rickstaa/stable-learning-control/commit/ba7088df6872fff630f872c615584327d3ae1281)
*   :bookmark: Updates code version to v0.9.4 [`6c1e72a`](https://github.com/rickstaa/stable-learning-control/commit/6c1e72ad70ecca4b6f569fb8aca536a2d8415f96)
*   :bookmark: Updates code version to v0.9.2 [`38d537a`](https://github.com/rickstaa/stable-learning-control/commit/38d537a1b18d31a900eb4af491c45f0dd0de4ab8)
*   :bookmark: Updates code version to v0.9.1 [`a4d218f`](https://github.com/rickstaa/stable-learning-control/commit/a4d218ff55d1187c32b69902b89bdf08b31799cf)
*   :wrench: Updates docs make file [`31217d4`](https://github.com/rickstaa/stable-learning-control/commit/31217d4077f5d27e9c42e24a791eef25c4a9ce06)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`99ff0d5`](https://github.com/rickstaa/stable-learning-control/commit/99ff0d5b6e5bca3b635604094bf823ff56be5e5e)
*   :bookmark: Updates code version to v0.9.3 [`91f5600`](https://github.com/rickstaa/stable-learning-control/commit/91f560080eda0cc1e909b2542d983e8611552739)
*   :fire: Removes redundant readme's [`76db499`](https://github.com/rickstaa/stable-learning-control/commit/76db49934dd7b774593a71572880445c5447901d)
*   :bulb: Updates code comments [`67ae469`](https://github.com/rickstaa/stable-learning-control/commit/67ae469e409b7a18b614973128c8682f4c78f564)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`711d62b`](https://github.com/rickstaa/stable-learning-control/commit/711d62b4bf29c3beff140dc5088ef3f51243ca15)
*   :alien: Updates simzoo submodule [`e79c619`](https://github.com/rickstaa/stable-learning-control/commit/e79c61985fde3e1bba635febdd3bd1ecefbb61d4)

## [v0.9.0](https://github.com/rickstaa/stable-learning-control/compare/v0.8.3...v0.9.0) - 2021-03-23

### Merged

*   :sparkles: Adds robustness eval utility [`#114`](https://github.com/rickstaa/stable-learning-control/pull/114)

### Commits

*   :memo: Updates CHANGELOG.md [`6f0893d`](https://github.com/rickstaa/stable-learning-control/commit/6f0893d4b7b0717313843613a17e1e7d134c6410)
*   :bookmark: Updates code version to v0.9.0 [`08204a5`](https://github.com/rickstaa/stable-learning-control/commit/08204a54691ee0ac61de43fc96bc964e7b692e82)
*   :bug: Fixed corrupt submodules [`0c76d7d`](https://github.com/rickstaa/stable-learning-control/commit/0c76d7d835f78765875480516d14e395ad926c5c)
*   :alien: Updates the simzoo submodule [`c13e770`](https://github.com/rickstaa/stable-learning-control/commit/c13e770c95b84782ce67bc71456057812208f8cf)
*   :wrench: Makes submodules work both with ssh and https [`96b54f0`](https://github.com/rickstaa/stable-learning-control/commit/96b54f017f2ae19fa62ed02e5664ce00f4c5fc9f)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`3fb9256`](https://github.com/rickstaa/stable-learning-control/commit/3fb925603c3df1286c8974b0a2c1e07577703fa0)
*   :building_construction: Fixed simzoo submodule setup [`100aeb2`](https://github.com/rickstaa/stable-learning-control/commit/100aeb255f9b424cc22567d4ff0ca8648f0a1475)

## [v0.8.3](https://github.com/rickstaa/stable-learning-control/compare/v0.8.2...v0.8.3) - 2021-03-11

### Commits

*   :memo: Updates CHANGELOG.md [`0574672`](https://github.com/rickstaa/stable-learning-control/commit/05746727b690f95aeb00e61bc0c40e6f033edfe2)
*   :sparkles: Adds checkpoint load ability to 'test_policy' utility [`59e28fe`](https://github.com/rickstaa/stable-learning-control/commit/59e28fe0ec5857790f6c6049229b601d85cb0281)
*   :bookmark: Updates code version to v0.8.3 [`3864255`](https://github.com/rickstaa/stable-learning-control/commit/386425557239960c939e5784d9f78c438e778b0f)

## [v0.8.2](https://github.com/rickstaa/stable-learning-control/compare/v0.8.1...v0.8.2) - 2021-03-11

### Commits

*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`9caeccb`](https://github.com/rickstaa/stable-learning-control/commit/9caeccb95b1147af2f7857d572c1463777e157ff)
*   :green_heart: Fixes apt update bug inside gh-actions [`782f2f9`](https://github.com/rickstaa/stable-learning-control/commit/782f2f9d0765ad60cd2088885be53c34ad8f31e1)

## [v0.8.1](https://github.com/rickstaa/stable-learning-control/compare/v0.8...v0.8.1) - 2021-03-11

## [v0.8](https://github.com/rickstaa/stable-learning-control/compare/v0.8.0...v0.8) - 2021-03-11

### Commits

*   :memo: Updates CHANGELOG.md [`336c15e`](https://github.com/rickstaa/stable-learning-control/commit/336c15e81f3c9bfab2f28601f8cc46075c8f9d46)
*   :memo: Updates CHANGELOG.md [`0574672`](https://github.com/rickstaa/stable-learning-control/commit/05746727b690f95aeb00e61bc0c40e6f033edfe2)
*   :sparkles: Adds checkpoint load ability to 'test_policy' utility [`59e28fe`](https://github.com/rickstaa/stable-learning-control/commit/59e28fe0ec5857790f6c6049229b601d85cb0281)
*   :bookmark: Updates code version to v0.8.3 [`3864255`](https://github.com/rickstaa/stable-learning-control/commit/386425557239960c939e5784d9f78c438e778b0f)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`9caeccb`](https://github.com/rickstaa/stable-learning-control/commit/9caeccb95b1147af2f7857d572c1463777e157ff)
*   :bug: Fixes logger checkpoint save bug [`ac4d3c2`](https://github.com/rickstaa/stable-learning-control/commit/ac4d3c2e53ad5b4f0cf4b1568f7585b8cd9f8795)
*   :memo: Updates documentation [`b2f09be`](https://github.com/rickstaa/stable-learning-control/commit/b2f09befb454e5d1ebf5507d3f73c2ff7a8673aa)
*   :green_heart: Fixes package naming inside the github-actions [`c8f080d`](https://github.com/rickstaa/stable-learning-control/commit/c8f080d967cb2ab85fc3edd8d3eb46f02931670c)
*   :green_heart: Fixes apt update bug inside gh-actions [`782f2f9`](https://github.com/rickstaa/stable-learning-control/commit/782f2f9d0765ad60cd2088885be53c34ad8f31e1)
*   :bookmark: Updates code version to v0.8.1 [`8656bdc`](https://github.com/rickstaa/stable-learning-control/commit/8656bdc5450f951bf9dc66447718797f67c62a5f)
*   :wrench: Changed docs latex engine to katex [`d00f989`](https://github.com/rickstaa/stable-learning-control/commit/d00f98992fb3055a96f73a49a1c6d7197d1c135e)

## [v0.8.0](https://github.com/rickstaa/stable-learning-control/compare/v0.7.1...v0.8.0) - 2021-03-10

### Commits

*   :truck: Rename repository to stable-learning-control [`a5024c9`](https://github.com/rickstaa/stable-learning-control/commit/a5024c936ccdea35d89dc924332107d2c0c851a4)

## [v0.7.1](https://github.com/rickstaa/stable-learning-control/compare/v0.7...v0.7.1) - 2021-03-06

### Commits

*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`c606392`](https://github.com/rickstaa/stable-learning-control/commit/c606392846f465afad1b86234de4ac2614fde3af)
*   :bug: Fixes a bug in the ray tuning example [`9bafe5d`](https://github.com/rickstaa/stable-learning-control/commit/9bafe5d744e3880f06dfbecb6c3e90c100c4db4d)

## [v0.7](https://github.com/rickstaa/stable-learning-control/compare/v0.7.0...v0.7) - 2021-03-06

## [v0.7.0](https://github.com/rickstaa/stable-learning-control/compare/v0.6...v0.7.0) - 2021-03-06

### Commits

*   :memo: Adds new --exp-cfg option to the docs [`9684404`](https://github.com/rickstaa/stable-learning-control/commit/96844047143571a111756214c5adae3ba177cd88)
*   :memo: Updates CHANGELOG.md [`fa8875f`](https://github.com/rickstaa/stable-learning-control/commit/fa8875fcc4e23f074655c682c74fd854f940a25f)
*   :sparkles: Adds --exp-cfg cmd-line argument [`e084ccf`](https://github.com/rickstaa/stable-learning-control/commit/e084ccf0f6bd3ae55762f1816aa061a5352a8a17)
*   :wrench: Updates --exp-cfg example configuration file [`983f9e5`](https://github.com/rickstaa/stable-learning-control/commit/983f9e59c5d480a10a62fdc5a24a5c37d28c66db)
*   :green_heart: Disables gh-actions on direct push to main [`d9c3fdf`](https://github.com/rickstaa/stable-learning-control/commit/d9c3fdf1a774aa757ff50e9e0c1f5001c60445ae)
*   :bookmark: Updates code version to v0.7.0 [`44b87ec`](https://github.com/rickstaa/stable-learning-control/commit/44b87ec35c385d50713c03a696c5ec381a1f108b)

## [v0.6](https://github.com/rickstaa/stable-learning-control/compare/v0.6.0...v0.6) - 2021-03-02

## [v0.6.0](https://github.com/rickstaa/stable-learning-control/compare/v0.5...v0.6.0) - 2021-03-02

### Merged

*   :memo: Update the documentation and cleans up the code [`#113`](https://github.com/rickstaa/stable-learning-control/pull/113)
*   :bug: Makes the test_policy utility compatible with tf2 [`#112`](https://github.com/rickstaa/stable-learning-control/pull/112)
*   :sparkles: Adds Torch GPU support [`#111`](https://github.com/rickstaa/stable-learning-control/pull/111)
*   :sparkles: Adds TF tensorboard support [`#110`](https://github.com/rickstaa/stable-learning-control/pull/110)
*   :bug: Fixes plotter [`#107`](https://github.com/rickstaa/stable-learning-control/pull/107)
*   :art: Format Python code with psf/black [`#106`](https://github.com/rickstaa/stable-learning-control/pull/106)
*   :bug: Fixes small learning rate lower bound bug [`#105`](https://github.com/rickstaa/stable-learning-control/pull/105)

### Commits

*   :memo: Adds API documentation [`16ed4f5`](https://github.com/rickstaa/stable-learning-control/commit/16ed4f5bc2366b0700048f7ffd8de9a5e5648b0b)
*   :sparkles: Adds torch tensorboard support [`885db1f`](https://github.com/rickstaa/stable-learning-control/commit/885db1f2c6437f514a2b7690379341b2033cdf9b)
*   :memo: Cleans up documentation [`cf97a97`](https://github.com/rickstaa/stable-learning-control/commit/cf97a97d2ac006fe704e2ee9997efebea74dfc19)
*   :sparkles: Adds custom gym environment support [`4d23b1e`](https://github.com/rickstaa/stable-learning-control/commit/4d23b1e9270abdbc2727e56606f7f4743ed2e6c7)
*   :memo: Updates CHANGELOG.md [`e97c34c`](https://github.com/rickstaa/stable-learning-control/commit/e97c34c52482cb65570ff6826d32aad60a995652)
*   :bug: Fixes plotter bug while cleaning up code [`cd8e2a2`](https://github.com/rickstaa/stable-learning-control/commit/cd8e2a291df13341d7ee758bde0e44f524d18031)
*   :wrench: Adds submodule pull check to the setup.py [`e313593`](https://github.com/rickstaa/stable-learning-control/commit/e3135930902dad06db8fb59e6aa9a851088a3e3a)
*   :sparkles: Adds a better method for loading custom gym environments [`f2e7d30`](https://github.com/rickstaa/stable-learning-control/commit/f2e7d309096e7c44a7c7919736411611210d406d)
*   :art: Cleans up plotter code [`2145e17`](https://github.com/rickstaa/stable-learning-control/commit/2145e1715d9c68b493087aa3d94058f35164a74d)
*   :bug: Fixes lr lower bound bug [`dff234e`](https://github.com/rickstaa/stable-learning-control/commit/dff234e34497a307dc25997c243798a4f14bf635)
*   :bookmark: Updates code version to v0.6.0 [`8f816e2`](https://github.com/rickstaa/stable-learning-control/commit/8f816e273c7cdeb1c1446b9ba43c813459a11b7e)
*   :alien: Updates simzoo submodule [`f8eff27`](https://github.com/rickstaa/stable-learning-control/commit/f8eff272befdbc46fe4be6ae01ad82ce33c518de)
*   :bug: Fixes small bug in test_policy utility [`de6a15a`](https://github.com/rickstaa/stable-learning-control/commit/de6a15a45d6c9b129115538da84fe1b33b782168)
*   :alien: Updates simzoo submodule [`798460d`](https://github.com/rickstaa/stable-learning-control/commit/798460d55326a92b81b84caeacffd95c3b82aef7)
*   :twisted_rightwards_arrows: Merge branch 'add_torch_tensorboard_support' into main [`8d324d1`](https://github.com/rickstaa/stable-learning-control/commit/8d324d1b4c0d7671b304514984f0347c4f59a2fe)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`9d30767`](https://github.com/rickstaa/stable-learning-control/commit/9d30767eb4ccc1d6c374b4aabd57ec03ae621e7f)
*   :wrench: Updates Torch hyperparameter defaults [`c8dd11e`](https://github.com/rickstaa/stable-learning-control/commit/c8dd11e038e5433a3463cceb24bfcb9cdcef7b08)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`3115913`](https://github.com/rickstaa/stable-learning-control/commit/3115913977a64f8648489050eb1d76ccf60f81c2)

## [v0.5](https://github.com/rickstaa/stable-learning-control/compare/v0.5.0...v0.5) - 2021-02-16

## [v0.5.0](https://github.com/rickstaa/stable-learning-control/compare/v0.4...v0.5.0) - 2021-02-16

### Merged

*   :art: Format Python code with psf/black [`#104`](https://github.com/rickstaa/stable-learning-control/pull/104)

### Commits

*   :memo: Updates CHANGELOG.md [`9a9533f`](https://github.com/rickstaa/stable-learning-control/commit/9a9533f56cdc6bd2e839cf70e6e78ee5ffc7f2f8)
*   :bookmark: Updates code version to v0.5.0 [`4b68c87`](https://github.com/rickstaa/stable-learning-control/commit/4b68c873cae1ffefdbafda3bef0e75dc89b9d27f)

## [v0.4](https://github.com/rickstaa/stable-learning-control/compare/v0.4.0...v0.4) - 2021-02-16

## [v0.4.0](https://github.com/rickstaa/stable-learning-control/compare/v0.3...v0.4.0) - 2021-02-16

### Merged

*   :arrow_up: Update dependency sphinx to v3.5.1 [`#101`](https://github.com/rickstaa/stable-learning-control/pull/101)
*   :sparkles: Add tf2 lac version [`#102`](https://github.com/rickstaa/stable-learning-control/pull/102)

### Commits

*   :memo: Updates CHANGELOG.md [`3dca15c`](https://github.com/rickstaa/stable-learning-control/commit/3dca15c87dafb30b113acc88fc65cb6f29777a15)
*   :bookmark: Updates code version to v0.4.0 [`da356fe`](https://github.com/rickstaa/stable-learning-control/commit/da356feb7c2fe590a64d9769453d40164cc16085)

## [v0.3](https://github.com/rickstaa/stable-learning-control/compare/v0.3.0...v0.3) - 2021-02-04

## [v0.3.0](https://github.com/rickstaa/stable-learning-control/compare/v0.2.11...v0.3.0) - 2021-02-04

### Merged

*   :arrow_up: Update dependency torch to v1.7.1 [`#58`](https://github.com/rickstaa/stable-learning-control/pull/58)
*   :sparkles: Adds torch lac algorithm [`#100`](https://github.com/rickstaa/stable-learning-control/pull/100)
*   :art: Format Python code with psf/black [`#95`](https://github.com/rickstaa/stable-learning-control/pull/95)
*   :art: Format Python code with psf/black [`#89`](https://github.com/rickstaa/stable-learning-control/pull/89)
*   :rotating_light: fix flake8 errors [`#88`](https://github.com/rickstaa/stable-learning-control/pull/88)
*   :art: Cleanup code [`#87`](https://github.com/rickstaa/stable-learning-control/pull/87)
*   :art: Format Python code with psf/black [`#86`](https://github.com/rickstaa/stable-learning-control/pull/86)
*   :art: Format Python code with psf/black [`#85`](https://github.com/rickstaa/stable-learning-control/pull/85)
*   :arrow_up: Update dependency sphinx to v3 [`#51`](https://github.com/rickstaa/stable-learning-control/pull/51)
*   :arrow_up: Update dependency sphinx-autobuild to v2020 [`#52`](https://github.com/rickstaa/stable-learning-control/pull/52)
*   :arrow_up: Update dependency sphinx-rtd-theme to v0.5.1 [`#65`](https://github.com/rickstaa/stable-learning-control/pull/65)
*   :pushpin: Pin dependency auto-changelog to 2.2.1 [`#84`](https://github.com/rickstaa/stable-learning-control/pull/84)

### Commits

*   :art: Cleans up code and removes redundant files [`ed84b13`](https://github.com/rickstaa/stable-learning-control/commit/ed84b133b7bec2ab3d0abb78e1f88883383e5ae8)
*   :memo: Updates CHANGELOG.md [`2a3b6bf`](https://github.com/rickstaa/stable-learning-control/commit/2a3b6bf9071f9a8b94e01133feb2bd930f2e64e2)
*   :art: Cleans up code structure [`bc7b485`](https://github.com/rickstaa/stable-learning-control/commit/bc7b485a85883453fa21fb823d4a8e7c478f5f68)
*   :memo: Updates CHANGELOG.md [`ae5ab82`](https://github.com/rickstaa/stable-learning-control/commit/ae5ab82a406be8751419d78d2dacee1f26684e6f)
*   :bookmark: Updates code version to v0.3.0 [`75c69c4`](https://github.com/rickstaa/stable-learning-control/commit/75c69c4f6500ffe823ca638bdaa5af493fc70f4b)
*   :bookmark: Updates documentation version to v0.2.11 [`021bbb6`](https://github.com/rickstaa/stable-learning-control/commit/021bbb686530067577479a7a5fc2e2cfad2d4de5)
*   :bug: Fixes a pip installation bug [`cd068c3`](https://github.com/rickstaa/stable-learning-control/commit/cd068c3b66add4831a87a683590a70fbb8bf24f0)
*   :art: Updates code formatting [`7248b7b`](https://github.com/rickstaa/stable-learning-control/commit/7248b7bd6ef18e40e552064d2900eaa85cee499e)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`f3928e3`](https://github.com/rickstaa/stable-learning-control/commit/f3928e35d64415a24b134ddd17da3fd61ea6816d)
*   :wrench: Updates release changelog configuration file [`9ff2884`](https://github.com/rickstaa/stable-learning-control/commit/9ff2884e8d68e2176e4e6f79754dabd2648aaa21)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`73c5494`](https://github.com/rickstaa/stable-learning-control/commit/73c54945e7683920ece91682697e4756f0850638)

## [v0.2.11](https://github.com/rickstaa/stable-learning-control/compare/v0.2.10...v0.2.11) - 2021-01-11

### Merged

*   :recycle: Cleanup code [`#83`](https://github.com/rickstaa/stable-learning-control/pull/83)

### Commits

*   :green_heart: Adds PAT to autotag action such that it triggers other workflows [`b505bd3`](https://github.com/rickstaa/stable-learning-control/commit/b505bd3bcfe6bb5794597ec5d45a8749942327ad)

## [v0.2.10](https://github.com/rickstaa/stable-learning-control/compare/v0.2.9...v0.2.10) - 2021-01-11

### Merged

*   :green_heart: Adds autotag on labeled pull request gh-action [`#82`](https://github.com/rickstaa/stable-learning-control/pull/82)

## [v0.2.9](https://github.com/rickstaa/stable-learning-control/compare/v0.2.8...v0.2.9) - 2021-01-11

### Commits

*   :memo: Adds auto pre-release github action [`fa304f3`](https://github.com/rickstaa/stable-learning-control/commit/fa304f3b8455bccc2f41f1b6065529bd6606c385)
*   :memo: Updates CHANGELOG.md [`5da98ef`](https://github.com/rickstaa/stable-learning-control/commit/5da98ef083c89e6d7225b97276333a133ffba10b)
*   :bookmark: Updates documentation version to v0.2.9 [`7fd9458`](https://github.com/rickstaa/stable-learning-control/commit/7fd945859ee8557f41de47d74e85b19665af71af)

## [v0.2.8](https://github.com/rickstaa/stable-learning-control/compare/v0.2.7...v0.2.8) - 2021-01-11

### Commits

*   :memo: Updates CHANGELOG.md [`733227a`](https://github.com/rickstaa/stable-learning-control/commit/733227a318b02edabdc0969d47b2267786ccaad8)
*   :bookmark: Updates documentation version to v0.2.8 [`9971c2d`](https://github.com/rickstaa/stable-learning-control/commit/9971c2d52915a93a7ed4907586644fb6f889cc95)
*   :green_heart: Fixes release action tagging behavoir [`33b459e`](https://github.com/rickstaa/stable-learning-control/commit/33b459e68b1c21e75c90ab32d00d0d08a8b23e71)
*   :memo: Updates CHANGELOG.md [`7285820`](https://github.com/rickstaa/stable-learning-control/commit/7285820b92f80843a9ac5b95aad654be4faea5f2)
*   :bookmark: Updates documentation version to v0.2.7 [`66c606d`](https://github.com/rickstaa/stable-learning-control/commit/66c606d39bef393e08571fa7b2e2275498dd8a41)

## [v0.2.7](https://github.com/rickstaa/stable-learning-control/compare/v0.2.6...v0.2.7) - 2021-01-11

### Commits

*   :green_heart: Fixes release gh-action [`abc7a33`](https://github.com/rickstaa/stable-learning-control/commit/abc7a33f61512fa257a827e2b60cb95fbf69a32f)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ca22707`](https://github.com/rickstaa/stable-learning-control/commit/ca227077eee983a10cd6f1ebd7f7df94986e8683)
*   :memo: Updates CHANGELOG.md [`3279843`](https://github.com/rickstaa/stable-learning-control/commit/32798433d45c5b12262d38b9d8f6612381300e50)
*   :green_heart: Fixes bug inside the release gh-action [`4c81a1e`](https://github.com/rickstaa/stable-learning-control/commit/4c81a1e522a17bc301128d0aa96104244617ca8b)
*   :bookmark: Updates documentation version to v0.2.6 [`43a55a5`](https://github.com/rickstaa/stable-learning-control/commit/43a55a59f0697e9c12b05b4d52c602344d572326)

## [v0.2.6](https://github.com/rickstaa/stable-learning-control/compare/v0.2.5...v0.2.6) - 2021-01-11

### Commits

*   :green_heart: Updates release github-action [`7bd6471`](https://github.com/rickstaa/stable-learning-control/commit/7bd6471703c27dc86f122508030b98d2024e90d7)
*   :green_heart: Fixes bug in release gh-action: [`da2db8a`](https://github.com/rickstaa/stable-learning-control/commit/da2db8a9ddc573e72d20442d0a984a3a49f9d1f6)

## [v0.2.5](https://github.com/rickstaa/stable-learning-control/compare/v0.2.4...v0.2.5) - 2021-01-10

### Commits

*   :green_heart: Fixes github release action [`20bc3e3`](https://github.com/rickstaa/stable-learning-control/commit/20bc3e3af866dda1b3e3e4e77e04370ab1a26ea7)
*   :memo: Updates CHANGELOG.md [`e5d2ea2`](https://github.com/rickstaa/stable-learning-control/commit/e5d2ea236be71f16e78b1e178ae3958dc6e2d969)
*   :bookmark: Updates documentation version to v0.2.4 [`7966ff5`](https://github.com/rickstaa/stable-learning-control/commit/7966ff5327943ac9cdcce8549b0e541e154f836e)

## [v0.2.4](https://github.com/rickstaa/stable-learning-control/compare/v0.2.3...v0.2.4) - 2021-01-10

### Commits

*   :green_heart: Fixes release gh action [`0dabacf`](https://github.com/rickstaa/stable-learning-control/commit/0dabacf78d409de0c5138f277ad04f20df1c08e3)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`9decccf`](https://github.com/rickstaa/stable-learning-control/commit/9decccf86248c292945fda3ac31bfbf505396adf)
*   :memo: Updates CHANGELOG.md [`a32d5d6`](https://github.com/rickstaa/stable-learning-control/commit/a32d5d6e0cf39501bbfbe705365e16ee74a74bdc)
*   :green_heart: Disables action-update semver [`cb05709`](https://github.com/rickstaa/stable-learning-control/commit/cb05709acacfcf250b2e9322739732c2f3080f39)
*   :bookmark: Updates documentation version to v0.2.3 [`5889c97`](https://github.com/rickstaa/stable-learning-control/commit/5889c971e395e5cb17fd42dcefdb1e57a8483a09)

## [v0.2.3](https://github.com/rickstaa/stable-learning-control/compare/v0.2.2...v0.2.3) - 2021-01-10

### Commits

*   :green_heart: Fixes release github action [`8a45522`](https://github.com/rickstaa/stable-learning-control/commit/8a45522844953cac1c6d461441966aca8a3d3a09)
*   :green_heart: Adds pre-release action [`745f4eb`](https://github.com/rickstaa/stable-learning-control/commit/745f4eb558c7217d3d6ee2d9c7bb4950473a5d97)
*   :memo: Updates CHANGELOG.md [`c0c06d8`](https://github.com/rickstaa/stable-learning-control/commit/c0c06d8b08d469670e1c5637a20020c4f9541054)
*   :bookmark: Updates documentation version to v0.2.2 [`c387560`](https://github.com/rickstaa/stable-learning-control/commit/c3875609a871ff8d3fa140be2623de421c96d9c1)
*   :green_heart: Fixes syntax error in release gh-action [`612e5ea`](https://github.com/rickstaa/stable-learning-control/commit/612e5ea117c9344771620f6da9c65eed481e7774)

## [v0.2.2](https://github.com/rickstaa/stable-learning-control/compare/v0.2.1...v0.2.2) - 2021-01-10

### Merged

*   :green_heart: Adds auto-changelog action [`#81`](https://github.com/rickstaa/stable-learning-control/pull/81)

### Commits

*   :bookmark: Updates documentation version to v0.2.0 [`d9fd0e2`](https://github.com/rickstaa/stable-learning-control/commit/d9fd0e29acec528dd1fa775b39aad51a6f7e153b)
*   :bookmark: Updates documentation version to v0.2.1 [`fbc8adc`](https://github.com/rickstaa/stable-learning-control/commit/fbc8adcc0af1a90aa764a509db699f843932f359)
*   :green_heart: Fixes changelog gh-action [`5f254d2`](https://github.com/rickstaa/stable-learning-control/commit/5f254d2a374f1c4df3c4b8794488af7044361508)

## [v0.2.1](https://github.com/rickstaa/stable-learning-control/compare/v0.2...v0.2.1) - 2021-01-10

### Commits

*   :green_heart: Fixes version release branch commit branch [`56b8f75`](https://github.com/rickstaa/stable-learning-control/commit/56b8f75f5d17a17c16dfc6490af4bfc32b398aed)

## [v0.2](https://github.com/rickstaa/stable-learning-control/compare/v0.2.0...v0.2) - 2021-01-11

### Merged

*   :recycle: Cleanup code [`#83`](https://github.com/rickstaa/stable-learning-control/pull/83)
*   :green_heart: Adds autotag on labeled pull request gh-action [`#82`](https://github.com/rickstaa/stable-learning-control/pull/82)
*   :green_heart: Adds auto-changelog action [`#81`](https://github.com/rickstaa/stable-learning-control/pull/81)
*   :white_check_mark: Test version release action [`#79`](https://github.com/rickstaa/stable-learning-control/pull/79)

### Commits

*   :green_heart: Updates release github-action [`7bd6471`](https://github.com/rickstaa/stable-learning-control/commit/7bd6471703c27dc86f122508030b98d2024e90d7)
*   :memo: Updates CHANGELOG.md [`32e735a`](https://github.com/rickstaa/stable-learning-control/commit/32e735a49ba2589201c08a9077e2da9cddffe953)
*   :green_heart: Fixes release github action [`8a45522`](https://github.com/rickstaa/stable-learning-control/commit/8a45522844953cac1c6d461441966aca8a3d3a09)
*   :green_heart: Adds pre-release action [`745f4eb`](https://github.com/rickstaa/stable-learning-control/commit/745f4eb558c7217d3d6ee2d9c7bb4950473a5d97)
*   :memo: Adds auto pre-release github action [`fa304f3`](https://github.com/rickstaa/stable-learning-control/commit/fa304f3b8455bccc2f41f1b6065529bd6606c385)
*   :green_heart: Fixes bug in release gh-action: [`da2db8a`](https://github.com/rickstaa/stable-learning-control/commit/da2db8a9ddc573e72d20442d0a984a3a49f9d1f6)
*   :memo: Updates CHANGELOG.md [`ae5ab82`](https://github.com/rickstaa/stable-learning-control/commit/ae5ab82a406be8751419d78d2dacee1f26684e6f)
*   :green_heart: Fixes github release action [`20bc3e3`](https://github.com/rickstaa/stable-learning-control/commit/20bc3e3af866dda1b3e3e4e77e04370ab1a26ea7)
*   :green_heart: Fixes release gh-action [`0d3be6a`](https://github.com/rickstaa/stable-learning-control/commit/0d3be6a8b7438c51af307e6f3d106af39317238b)
*   :memo: Updates CHANGELOG.md [`c0c06d8`](https://github.com/rickstaa/stable-learning-control/commit/c0c06d8b08d469670e1c5637a20020c4f9541054)
*   :memo: Updates CHANGELOG.md [`733227a`](https://github.com/rickstaa/stable-learning-control/commit/733227a318b02edabdc0969d47b2267786ccaad8)
*   :memo: Updates CHANGELOG.md [`e5d2ea2`](https://github.com/rickstaa/stable-learning-control/commit/e5d2ea236be71f16e78b1e178ae3958dc6e2d969)
*   :memo: Updates CHANGELOG.md [`a32d5d6`](https://github.com/rickstaa/stable-learning-control/commit/a32d5d6e0cf39501bbfbe705365e16ee74a74bdc)
*   :memo: Updates CHANGELOG.md [`3279843`](https://github.com/rickstaa/stable-learning-control/commit/32798433d45c5b12262d38b9d8f6612381300e50)
*   :green_heart: Fixes bug inside the release gh-action [`4c81a1e`](https://github.com/rickstaa/stable-learning-control/commit/4c81a1e522a17bc301128d0aa96104244617ca8b)
*   :green_heart: Updates gh-actions [`f1a7a71`](https://github.com/rickstaa/stable-learning-control/commit/f1a7a71b7c5186684aba5ef9a61e32b09d8da6d0)
*   :green_heart: Disables action-update semver [`cb05709`](https://github.com/rickstaa/stable-learning-control/commit/cb05709acacfcf250b2e9322739732c2f3080f39)
*   :bookmark: Updates documentation version to v0.2.11 [`021bbb6`](https://github.com/rickstaa/stable-learning-control/commit/021bbb686530067577479a7a5fc2e2cfad2d4de5)
*   :memo: Updates CHANGELOG.md [`5da98ef`](https://github.com/rickstaa/stable-learning-control/commit/5da98ef083c89e6d7225b97276333a133ffba10b)
*   :green_heart: Fixes release gh-action [`abc7a33`](https://github.com/rickstaa/stable-learning-control/commit/abc7a33f61512fa257a827e2b60cb95fbf69a32f)
*   :bookmark: Updates documentation version to v0.2.9 [`7fd9458`](https://github.com/rickstaa/stable-learning-control/commit/7fd945859ee8557f41de47d74e85b19665af71af)
*   :bookmark: Updates documentation version to v0.2.8 [`9971c2d`](https://github.com/rickstaa/stable-learning-control/commit/9971c2d52915a93a7ed4907586644fb6f889cc95)
*   :green_heart: Fixes release action tagging behavoir [`33b459e`](https://github.com/rickstaa/stable-learning-control/commit/33b459e68b1c21e75c90ab32d00d0d08a8b23e71)
*   :memo: Updates CHANGELOG.md [`7285820`](https://github.com/rickstaa/stable-learning-control/commit/7285820b92f80843a9ac5b95aad654be4faea5f2)
*   :bookmark: Updates documentation version to v0.2.7 [`66c606d`](https://github.com/rickstaa/stable-learning-control/commit/66c606d39bef393e08571fa7b2e2275498dd8a41)
*   :bookmark: Updates documentation version to v0.2.6 [`43a55a5`](https://github.com/rickstaa/stable-learning-control/commit/43a55a59f0697e9c12b05b4d52c602344d572326)
*   :bookmark: Updates documentation version to v0.2.4 [`7966ff5`](https://github.com/rickstaa/stable-learning-control/commit/7966ff5327943ac9cdcce8549b0e541e154f836e)
*   :bookmark: Updates documentation version to v0.2.0 [`d9fd0e2`](https://github.com/rickstaa/stable-learning-control/commit/d9fd0e29acec528dd1fa775b39aad51a6f7e153b)
*   :bookmark: Updates documentation version to v0.2.1 [`fbc8adc`](https://github.com/rickstaa/stable-learning-control/commit/fbc8adcc0af1a90aa764a509db699f843932f359)
*   :green_heart: Adds PAT to autotag action such that it triggers other workflows [`b505bd3`](https://github.com/rickstaa/stable-learning-control/commit/b505bd3bcfe6bb5794597ec5d45a8749942327ad)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ca22707`](https://github.com/rickstaa/stable-learning-control/commit/ca227077eee983a10cd6f1ebd7f7df94986e8683)
*   :bookmark: Updates documentation version to v0.2.2 [`c387560`](https://github.com/rickstaa/stable-learning-control/commit/c3875609a871ff8d3fa140be2623de421c96d9c1)
*   :green_heart: Fixes syntax error in release gh-action [`612e5ea`](https://github.com/rickstaa/stable-learning-control/commit/612e5ea117c9344771620f6da9c65eed481e7774)
*   :green_heart: Fixes release gh action [`0dabacf`](https://github.com/rickstaa/stable-learning-control/commit/0dabacf78d409de0c5138f277ad04f20df1c08e3)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`9decccf`](https://github.com/rickstaa/stable-learning-control/commit/9decccf86248c292945fda3ac31bfbf505396adf)
*   :bookmark: Updates documentation version to v0.2.3 [`5889c97`](https://github.com/rickstaa/stable-learning-control/commit/5889c971e395e5cb17fd42dcefdb1e57a8483a09)
*   :green_heart: Fixes changelog gh-action [`5f254d2`](https://github.com/rickstaa/stable-learning-control/commit/5f254d2a374f1c4df3c4b8794488af7044361508)
*   :green_heart: Fixes version release job [`4c0011f`](https://github.com/rickstaa/stable-learning-control/commit/4c0011f84ecbeba795cbf8f460edfd2bf7979565)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`18afc55`](https://github.com/rickstaa/stable-learning-control/commit/18afc554d876406bc67a357343b6c3e5499cb409)

## [v0.2.0](https://github.com/rickstaa/stable-learning-control/compare/v0.1.9...v0.2.0) - 2021-01-10

### Merged

*   :bug: Fixes bug in release gh-action [`#78`](https://github.com/rickstaa/stable-learning-control/pull/78)
*   :green_heart: Ci/gh actions [`#77`](https://github.com/rickstaa/stable-learning-control/pull/77)

### Commits

*   :memo: Updates CHANGELOG.md [`ed54568`](https://github.com/rickstaa/stable-learning-control/commit/ed545689e69ac03315d368a27183cdced7259cc0)
*   :green_heart: Fixes release workflow [`8d7b851`](https://github.com/rickstaa/stable-learning-control/commit/8d7b85195531436abca126d618421edf234fc576)

## [v0.1.9](https://github.com/rickstaa/stable-learning-control/compare/v0.1.8...v0.1.9) - 2021-01-10

## [v0.1.8](https://github.com/rickstaa/stable-learning-control/compare/v0.1.7...v0.1.8) - 2021-01-10

## [v0.1.7](https://github.com/rickstaa/stable-learning-control/compare/v0.1.6...v0.1.7) - 2021-01-10

### Merged

*   :green_heart: Seperates the release gh-action into seperate jobs [`#76`](https://github.com/rickstaa/stable-learning-control/pull/76)

### Commits

*   :bookmark: Updates documentation version to v0.1.6 [`26032b2`](https://github.com/rickstaa/stable-learning-control/commit/26032b2ea88ec35cf3e532da8f7ae267d31cab00)

## [v0.1.6](https://github.com/rickstaa/stable-learning-control/compare/v0.1.5...v0.1.6) - 2021-01-09

### Merged

*   :green_heart: Fix gh actions [`#75`](https://github.com/rickstaa/stable-learning-control/pull/75)
*   :green_heart: fix gh actions [`#74`](https://github.com/rickstaa/stable-learning-control/pull/74)
*   :green_heart: Updates mac github actions [`#72`](https://github.com/rickstaa/stable-learning-control/pull/72)

### Commits

*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`f231c13`](https://github.com/rickstaa/stable-learning-control/commit/f231c131cfa3b46d5176b2679721235786453b6f)
*   :memo: Updates CHANGELOG.md [`51a3492`](https://github.com/rickstaa/stable-learning-control/commit/51a349204521c50f483ec4971ee824b446c931cf)
*   :bookmark: Updates documentation version to v0.1.5 [`80aee7a`](https://github.com/rickstaa/stable-learning-control/commit/80aee7a1e7f78579c8da8a604d383e08da56bdc1)
*   :bug: Fixes mac github action name [`ca563f7`](https://github.com/rickstaa/stable-learning-control/commit/ca563f7b2d52b86fab4b7edc8be22e53e1b780ef)

## [v0.1.5](https://github.com/rickstaa/stable-learning-control/compare/v0.1.4...v0.1.5) - 2021-01-09

### Merged

*   :art: Format Python code with psf/black [`#70`](https://github.com/rickstaa/stable-learning-control/pull/70)
*   :art: Format Python code with psf/black [`#69`](https://github.com/rickstaa/stable-learning-control/pull/69)
*   :art: Format markdown code with remark-lint [`#68`](https://github.com/rickstaa/stable-learning-control/pull/68)

### Commits

*   :rewind: Revert ":bookmark: Updates documentation version: v0.1.5 → v0.1.5" [`284f438`](https://github.com/rickstaa/stable-learning-control/commit/284f4381d4bb36c82be653f59490dacea02ff947)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6104eef`](https://github.com/rickstaa/stable-learning-control/commit/6104eefb1d1f5a91989f4457d86b2bce6732cb55)
*   :memo: Updates the documentation [`7544d75`](https://github.com/rickstaa/stable-learning-control/commit/7544d75c4583ae94f64daa32b420338d40ac47d9)
*   :green_heart: Fixes bug in release gh-action [`6067635`](https://github.com/rickstaa/stable-learning-control/commit/60676351b3821c4ed6be725e2570209a21562788)
*   :green_heart: Updates docs and release gh actions [`3cff977`](https://github.com/rickstaa/stable-learning-control/commit/3cff9779bf653d542515170804418ae20619d339)
*   :fire: Removes gh-action test data [`135646b`](https://github.com/rickstaa/stable-learning-control/commit/135646b898fe957945af1a625ea5ac82c6089ef7)
*   :goal_net: Catch errors to allow autoformatting [`4c78b41`](https://github.com/rickstaa/stable-learning-control/commit/4c78b414a6126cbe15eb5c1d8c7074aaffbb477f)
*   :green_heart: Updates gh-action [`07e36b7`](https://github.com/rickstaa/stable-learning-control/commit/07e36b7d62ef87f8ffbd85b5cc691d58d4ee5c35)
*   :green_heart: Updates code quality gh-actions [`16591b4`](https://github.com/rickstaa/stable-learning-control/commit/16591b43623ea2fc3eca0dc37c171540203fc064)
*   :green_heart: Fixes release gh-action [`5e1d376`](https://github.com/rickstaa/stable-learning-control/commit/5e1d376a05950c1d519585c551b86df58da8849b)
*   :green_heart: Updates release gh-action [`f966ee1`](https://github.com/rickstaa/stable-learning-control/commit/f966ee1044b137754c72ba2b15f2327d3e5cbdab)
*   :green_heart: Fixes output bug in release gh-action [`59cc86f`](https://github.com/rickstaa/stable-learning-control/commit/59cc86fcdbb5e0e1d186924725903db3a01eed86)
*   :green_heart: Fixes release github action [`22830dd`](https://github.com/rickstaa/stable-learning-control/commit/22830dde021962239bbce9c31c36f0dc77354000)
*   :green_heart: Fixes gh release action detached head bug [`69471ac`](https://github.com/rickstaa/stable-learning-control/commit/69471acf89bf9e3d50ae06559f1b215ed1020094)
*   :bookmark: Updates documentation version: v0.1.5 → v0.1.5 [`06dce19`](https://github.com/rickstaa/stable-learning-control/commit/06dce19b50cf3bdd8dcc1cee4595ea5986ceef8b)
*   :bookmark: Updates documentation versioning [`583393c`](https://github.com/rickstaa/stable-learning-control/commit/583393c413e63df64e370d8b45f2eff9b3e56751)
*   :green_heart: Updates code quality gh-action syntax [`14497ad`](https://github.com/rickstaa/stable-learning-control/commit/14497ad148c1a5a0fb9eb8d285b68fa973c7b43a)
*   :green_heart: Fixes bug in release gh-action [`483ad95`](https://github.com/rickstaa/stable-learning-control/commit/483ad95a3af8dce729f86cca9c2ca76323a6bd4c)
*   :green_heart: Fixes release gh-action commit step [`a9dfb68`](https://github.com/rickstaa/stable-learning-control/commit/a9dfb68276c0b0c5c486bf888c5ea1e064e33c1b)
*   :green_heart: Fixes release gh-action [`3578002`](https://github.com/rickstaa/stable-learning-control/commit/3578002e09103e93fb3f16b37963b4cd3d75cb57)
*   :green_heart: Fixes gh release action branch name [`84dad7e`](https://github.com/rickstaa/stable-learning-control/commit/84dad7ebc95a6233face3d9d60f51f5c128c953f)
*   :green_heart: Fixes release gh-action [`1f68e88`](https://github.com/rickstaa/stable-learning-control/commit/1f68e8851554e84582cccc634830f2a269091661)
*   :green_heart: Fixes other bug in release github-action [`5dfa2de`](https://github.com/rickstaa/stable-learning-control/commit/5dfa2de0212608b5518c066d410772b05248f8c5)
*   :green_heart: Fixes release gh-action [`41191c6`](https://github.com/rickstaa/stable-learning-control/commit/41191c634033c7675eb72c8949cd9831dbf64131)
*   :green_heart: Updates release gh-action [`bf43239`](https://github.com/rickstaa/stable-learning-control/commit/bf43239413fa439b31b77c2e29bc1561977416d7)
*   :memo: Updates documentation [`960da39`](https://github.com/rickstaa/stable-learning-control/commit/960da398e0a794e19e2646d79ee8b13e58491412)
*   :memo: Updates documentation [`1a89225`](https://github.com/rickstaa/stable-learning-control/commit/1a892255037ae355748bb94fcbcc91e03c00c28d)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ac3bcd8`](https://github.com/rickstaa/stable-learning-control/commit/ac3bcd8bd4dabd8f3e3700b48e38ee314a588a87)

## [v0.1.4](https://github.com/rickstaa/stable-learning-control/compare/v0.1.3...v0.1.4) - 2020-12-24

### Commits

*   :green_heart: Updates mlc release and docs release gh-actions [`1b10a48`](https://github.com/rickstaa/stable-learning-control/commit/1b10a48ddde5d3fba96dd91033b9dae120ecc0bb)

## [v0.1.3](https://github.com/rickstaa/stable-learning-control/compare/v0.1.2...v0.1.3) - 2020-12-24

### Commits

*   :green_heart: Fixes release gh action [`6913c08`](https://github.com/rickstaa/stable-learning-control/commit/6913c089a45936f3181775e6cb2fbbf6a4cce59d)

## [v0.1.2](https://github.com/rickstaa/stable-learning-control/compare/v0.1.1...v0.1.2) - 2020-12-24

### Merged

*   :alien: Updates simzoo submodule [`#62`](https://github.com/rickstaa/stable-learning-control/pull/62)
*   :green_heart: Fixes gh-actions [`#61`](https://github.com/rickstaa/stable-learning-control/pull/61)

### Commits

*   :green_heart: Adds gh-action test data [`840bd04`](https://github.com/rickstaa/stable-learning-control/commit/840bd0415f3a558d3734c601b2508f563924cf4d)
*   :art: Format Python code with psf/black push [`c512828`](https://github.com/rickstaa/stable-learning-control/commit/c512828394574a103c106a89d7e4455fe4bcfa63)
*   :green_heart: Adds auto-changelog and bumpversion gh-actions [`f79e7db`](https://github.com/rickstaa/stable-learning-control/commit/f79e7db5ba0768c4405d7daf358b8bcfffd8a097)
*   :green_heart: Fixes quality check gh-action [`0fb043a`](https://github.com/rickstaa/stable-learning-control/commit/0fb043a9238c154656f88d1b31afb9ce68a7b119)
*   :green_heart: Enables code quality gh-actions [`2e1ed92`](https://github.com/rickstaa/stable-learning-control/commit/2e1ed92459739321670d7c48e2548df9877031f2)
*   :green_heart: Updates github actions [`76a9f86`](https://github.com/rickstaa/stable-learning-control/commit/76a9f86ead050d7c19700e76959a51522836b7c6)
*   :memo: Updates Changelog [`de02746`](https://github.com/rickstaa/stable-learning-control/commit/de027469761da81d7702622bbf82eea12d206527)
*   :green_heart: Update docs check github action. [`d293b42`](https://github.com/rickstaa/stable-learning-control/commit/d293b425702d758c00dc7bf8559442f5bc92f041)
*   :green_heart: Updates docs gh action [`3538689`](https://github.com/rickstaa/stable-learning-control/commit/35386898d75341969a56d6f31841347054e12335)
*   :white_check_mark: Test out pull request creation action [`9bb87d2`](https://github.com/rickstaa/stable-learning-control/commit/9bb87d2f812ae4406be6478c10932058d0305307)
*   :green_heart: Updates gh-actions [`78c476a`](https://github.com/rickstaa/stable-learning-control/commit/78c476a26c48528bb59f09f9e10923d537266992)
*   :bug: Updates python version inside the gh-actions [`8ffa382`](https://github.com/rickstaa/stable-learning-control/commit/8ffa3826bc1609c2924af8b2c39ded3add391375)
*   :green_heart: Fixes release gh action [`c03c72b`](https://github.com/rickstaa/stable-learning-control/commit/c03c72bf5b01dc893ee4eb12eb93b9defbb6b97d)
*   :bug: Fixes bug in gh docs publish workflow [`234e2b8`](https://github.com/rickstaa/stable-learning-control/commit/234e2b85feb6839fdfa26b15fdadbc1ba77b15d9)
*   Fixes syntax bug in gh-action [`8501fb6`](https://github.com/rickstaa/stable-learning-control/commit/8501fb6d94453305770856d216f74238c0d67696)
*   :memo: Change README.md buttons [`8041819`](https://github.com/rickstaa/stable-learning-control/commit/8041819a5595b2e3b522c73a9ec83f9ca3c954da)
*   :green_heart: Fixes syntax error inside github action [`18728b8`](https://github.com/rickstaa/stable-learning-control/commit/18728b84dbe6f096f83d07818ee9556c6de12bf9)
*   :white_check_mark: Enables pre-release workflow [`6054e7d`](https://github.com/rickstaa/stable-learning-control/commit/6054e7de78d0997e00c5574f779fd53ad9e8b571)
*   :green_heart: Force the python cache to be rebuild [`d6172d5`](https://github.com/rickstaa/stable-learning-control/commit/d6172d5f7c2892a7fdbf07e18f0d6c116f6f82cb)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`b20f329`](https://github.com/rickstaa/stable-learning-control/commit/b20f329875df594928636277370328f5a5192dd6)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6e6f3c8`](https://github.com/rickstaa/stable-learning-control/commit/6e6f3c8a3acdd01222541dc29164e771224ca975)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`793cf6a`](https://github.com/rickstaa/stable-learning-control/commit/793cf6a70514c5926ad647375790f65ff7302316)

## [v0.1.1](https://github.com/rickstaa/stable-learning-control/compare/v0.1...v0.1.1) - 2020-12-22

## [v0.1](https://github.com/rickstaa/stable-learning-control/compare/v0.1.0...v0.1) - 2021-01-10

### Merged

*   :bug: Fixes bug in release gh-action [`#78`](https://github.com/rickstaa/stable-learning-control/pull/78)
*   :green_heart: Ci/gh actions [`#77`](https://github.com/rickstaa/stable-learning-control/pull/77)
*   :green_heart: Seperates the release gh-action into seperate jobs [`#76`](https://github.com/rickstaa/stable-learning-control/pull/76)
*   :green_heart: Fix gh actions [`#75`](https://github.com/rickstaa/stable-learning-control/pull/75)
*   :green_heart: fix gh actions [`#74`](https://github.com/rickstaa/stable-learning-control/pull/74)
*   :green_heart: Updates mac github actions [`#72`](https://github.com/rickstaa/stable-learning-control/pull/72)
*   :art: Format Python code with psf/black [`#70`](https://github.com/rickstaa/stable-learning-control/pull/70)
*   :art: Format Python code with psf/black [`#69`](https://github.com/rickstaa/stable-learning-control/pull/69)
*   :art: Format markdown code with remark-lint [`#68`](https://github.com/rickstaa/stable-learning-control/pull/68)
*   :alien: Updates simzoo submodule [`#62`](https://github.com/rickstaa/stable-learning-control/pull/62)
*   :green_heart: Fixes gh-actions [`#61`](https://github.com/rickstaa/stable-learning-control/pull/61)

### Commits

*   :memo: Updates the documentation [`7544d75`](https://github.com/rickstaa/stable-learning-control/commit/7544d75c4583ae94f64daa32b420338d40ac47d9)
*   :memo: Updates CHANGELOG.md [`ed54568`](https://github.com/rickstaa/stable-learning-control/commit/ed545689e69ac03315d368a27183cdced7259cc0)
*   :bookmark: Updates documentation version to v0.1.6 [`26032b2`](https://github.com/rickstaa/stable-learning-control/commit/26032b2ea88ec35cf3e532da8f7ae267d31cab00)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`f231c13`](https://github.com/rickstaa/stable-learning-control/commit/f231c131cfa3b46d5176b2679721235786453b6f)
*   :green_heart: Fixes bug in release gh-action [`6067635`](https://github.com/rickstaa/stable-learning-control/commit/60676351b3821c4ed6be725e2570209a21562788)
*   :green_heart: Updates docs and release gh actions [`3cff977`](https://github.com/rickstaa/stable-learning-control/commit/3cff9779bf653d542515170804418ae20619d339)
*   :fire: Removes gh-action test data [`135646b`](https://github.com/rickstaa/stable-learning-control/commit/135646b898fe957945af1a625ea5ac82c6089ef7)
*   :memo: Updates CHANGELOG.md [`51a3492`](https://github.com/rickstaa/stable-learning-control/commit/51a349204521c50f483ec4971ee824b446c931cf)
*   :green_heart: Updates gh-action [`07e36b7`](https://github.com/rickstaa/stable-learning-control/commit/07e36b7d62ef87f8ffbd85b5cc691d58d4ee5c35)
*   :green_heart: Updates code quality gh-actions [`16591b4`](https://github.com/rickstaa/stable-learning-control/commit/16591b43623ea2fc3eca0dc37c171540203fc064)
*   :green_heart: Fixes release gh-action [`5e1d376`](https://github.com/rickstaa/stable-learning-control/commit/5e1d376a05950c1d519585c551b86df58da8849b)
*   :green_heart: Updates release gh-action [`f966ee1`](https://github.com/rickstaa/stable-learning-control/commit/f966ee1044b137754c72ba2b15f2327d3e5cbdab)
*   :green_heart: Fixes output bug in release gh-action [`59cc86f`](https://github.com/rickstaa/stable-learning-control/commit/59cc86fcdbb5e0e1d186924725903db3a01eed86)
*   :green_heart: Fixes release github action [`22830dd`](https://github.com/rickstaa/stable-learning-control/commit/22830dde021962239bbce9c31c36f0dc77354000)
*   :green_heart: Fixes gh release action detached head bug [`69471ac`](https://github.com/rickstaa/stable-learning-control/commit/69471acf89bf9e3d50ae06559f1b215ed1020094)
*   :bookmark: Updates documentation version to v0.1.5 [`80aee7a`](https://github.com/rickstaa/stable-learning-control/commit/80aee7a1e7f78579c8da8a604d383e08da56bdc1)
*   :rewind: Revert ":bookmark: Updates documentation version: v0.1.5 → v0.1.5" [`284f438`](https://github.com/rickstaa/stable-learning-control/commit/284f4381d4bb36c82be653f59490dacea02ff947)
*   :bookmark: Updates documentation version: v0.1.5 → v0.1.5 [`06dce19`](https://github.com/rickstaa/stable-learning-control/commit/06dce19b50cf3bdd8dcc1cee4595ea5986ceef8b)
*   :bookmark: Updates documentation versioning [`583393c`](https://github.com/rickstaa/stable-learning-control/commit/583393c413e63df64e370d8b45f2eff9b3e56751)
*   :green_heart: Updates code quality gh-action syntax [`14497ad`](https://github.com/rickstaa/stable-learning-control/commit/14497ad148c1a5a0fb9eb8d285b68fa973c7b43a)
*   :green_heart: Fixes bug in release gh-action [`483ad95`](https://github.com/rickstaa/stable-learning-control/commit/483ad95a3af8dce729f86cca9c2ca76323a6bd4c)
*   :green_heart: Fixes release gh-action commit step [`a9dfb68`](https://github.com/rickstaa/stable-learning-control/commit/a9dfb68276c0b0c5c486bf888c5ea1e064e33c1b)
*   :green_heart: Fixes release gh-action [`3578002`](https://github.com/rickstaa/stable-learning-control/commit/3578002e09103e93fb3f16b37963b4cd3d75cb57)
*   :bug: Fixes mac github action name [`ca563f7`](https://github.com/rickstaa/stable-learning-control/commit/ca563f7b2d52b86fab4b7edc8be22e53e1b780ef)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6104eef`](https://github.com/rickstaa/stable-learning-control/commit/6104eefb1d1f5a91989f4457d86b2bce6732cb55)
*   :green_heart: Fixes gh release action branch name [`84dad7e`](https://github.com/rickstaa/stable-learning-control/commit/84dad7ebc95a6233face3d9d60f51f5c128c953f)
*   :green_heart: Fixes release gh-action [`1f68e88`](https://github.com/rickstaa/stable-learning-control/commit/1f68e8851554e84582cccc634830f2a269091661)
*   :green_heart: Fixes other bug in release github-action [`5dfa2de`](https://github.com/rickstaa/stable-learning-control/commit/5dfa2de0212608b5518c066d410772b05248f8c5)
*   :green_heart: Fixes release gh-action [`41191c6`](https://github.com/rickstaa/stable-learning-control/commit/41191c634033c7675eb72c8949cd9831dbf64131)
*   :green_heart: Updates release gh-action [`bf43239`](https://github.com/rickstaa/stable-learning-control/commit/bf43239413fa439b31b77c2e29bc1561977416d7)
*   :memo: Updates documentation [`960da39`](https://github.com/rickstaa/stable-learning-control/commit/960da398e0a794e19e2646d79ee8b13e58491412)
*   :memo: Updates documentation [`1a89225`](https://github.com/rickstaa/stable-learning-control/commit/1a892255037ae355748bb94fcbcc91e03c00c28d)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ac3bcd8`](https://github.com/rickstaa/stable-learning-control/commit/ac3bcd8bd4dabd8f3e3700b48e38ee314a588a87)
*   :green_heart: Adds gh-action test data [`840bd04`](https://github.com/rickstaa/stable-learning-control/commit/840bd0415f3a558d3734c601b2508f563924cf4d)
*   :art: Format Python code with psf/black push [`c512828`](https://github.com/rickstaa/stable-learning-control/commit/c512828394574a103c106a89d7e4455fe4bcfa63)
*   :green_heart: Adds auto-changelog and bumpversion gh-actions [`f79e7db`](https://github.com/rickstaa/stable-learning-control/commit/f79e7db5ba0768c4405d7daf358b8bcfffd8a097)
*   :green_heart: Fixes quality check gh-action [`0fb043a`](https://github.com/rickstaa/stable-learning-control/commit/0fb043a9238c154656f88d1b31afb9ce68a7b119)
*   :goal_net: Catch errors to allow autoformatting [`4c78b41`](https://github.com/rickstaa/stable-learning-control/commit/4c78b414a6126cbe15eb5c1d8c7074aaffbb477f)
*   :green_heart: Enables code quality gh-actions [`2e1ed92`](https://github.com/rickstaa/stable-learning-control/commit/2e1ed92459739321670d7c48e2548df9877031f2)
*   :green_heart: Updates github actions [`76a9f86`](https://github.com/rickstaa/stable-learning-control/commit/76a9f86ead050d7c19700e76959a51522836b7c6)
*   :memo: Updates Changelog [`de02746`](https://github.com/rickstaa/stable-learning-control/commit/de027469761da81d7702622bbf82eea12d206527)
*   :green_heart: Update docs check github action. [`d293b42`](https://github.com/rickstaa/stable-learning-control/commit/d293b425702d758c00dc7bf8559442f5bc92f041)
*   :green_heart: Updates docs gh action [`3538689`](https://github.com/rickstaa/stable-learning-control/commit/35386898d75341969a56d6f31841347054e12335)
*   :white_check_mark: Test out pull request creation action [`9bb87d2`](https://github.com/rickstaa/stable-learning-control/commit/9bb87d2f812ae4406be6478c10932058d0305307)
*   :green_heart: Updates gh-actions [`78c476a`](https://github.com/rickstaa/stable-learning-control/commit/78c476a26c48528bb59f09f9e10923d537266992)
*   :memo: Updates bug_report.md [`aeb5e9b`](https://github.com/rickstaa/stable-learning-control/commit/aeb5e9becee401399ae1272ab762d5b59c44360b)
*   :bug: Updates python version inside the gh-actions [`8ffa382`](https://github.com/rickstaa/stable-learning-control/commit/8ffa3826bc1609c2924af8b2c39ded3add391375)
*   :green_heart: Fixes release gh action [`c03c72b`](https://github.com/rickstaa/stable-learning-control/commit/c03c72bf5b01dc893ee4eb12eb93b9defbb6b97d)
*   :bookmark: Bump version: 0.1.0 → 0.1.1 [`f25db48`](https://github.com/rickstaa/stable-learning-control/commit/f25db48a9114dbc5db50f26f43f2c8ab242db9d0)
*   :memo: Updates the os badge urls [`33546b1`](https://github.com/rickstaa/stable-learning-control/commit/33546b17bec3e3757d414726e1017ca05c31122a)
*   :green_heart: Fixes release gh action [`6913c08`](https://github.com/rickstaa/stable-learning-control/commit/6913c089a45936f3181775e6cb2fbbf6a4cce59d)
*   :bug: Fixes bug in gh docs publish workflow [`234e2b8`](https://github.com/rickstaa/stable-learning-control/commit/234e2b85feb6839fdfa26b15fdadbc1ba77b15d9)
*   :green_heart: Updates mlc release and docs release gh-actions [`1b10a48`](https://github.com/rickstaa/stable-learning-control/commit/1b10a48ddde5d3fba96dd91033b9dae120ecc0bb)
*   Fixes syntax bug in gh-action [`8501fb6`](https://github.com/rickstaa/stable-learning-control/commit/8501fb6d94453305770856d216f74238c0d67696)
*   :memo: Change README.md buttons [`8041819`](https://github.com/rickstaa/stable-learning-control/commit/8041819a5595b2e3b522c73a9ec83f9ca3c954da)
*   :green_heart: Fixes syntax error inside github action [`18728b8`](https://github.com/rickstaa/stable-learning-control/commit/18728b84dbe6f096f83d07818ee9556c6de12bf9)
*   :white_check_mark: Enables pre-release workflow [`6054e7d`](https://github.com/rickstaa/stable-learning-control/commit/6054e7de78d0997e00c5574f779fd53ad9e8b571)
*   :bug: Fixes a small bug in the docs release ci action [`6548c52`](https://github.com/rickstaa/stable-learning-control/commit/6548c5249521e0c73aff2241761e666c08913254)
*   :green_heart: Force the python cache to be rebuild [`d6172d5`](https://github.com/rickstaa/stable-learning-control/commit/d6172d5f7c2892a7fdbf07e18f0d6c116f6f82cb)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`b20f329`](https://github.com/rickstaa/stable-learning-control/commit/b20f329875df594928636277370328f5a5192dd6)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6e6f3c8`](https://github.com/rickstaa/stable-learning-control/commit/6e6f3c8a3acdd01222541dc29164e771224ca975)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`793cf6a`](https://github.com/rickstaa/stable-learning-control/commit/793cf6a70514c5926ad647375790f65ff7302316)
*   :construction_worker: Changes the documentation ci name [`80c9049`](https://github.com/rickstaa/stable-learning-control/commit/80c9049c3058c1245997d4ec9b0b74a25b6e8524)
*   :bug: Fixes a small ci bug [`5cb1437`](https://github.com/rickstaa/stable-learning-control/commit/5cb14375f76cfec82ac390a5b6616b744c607319)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`4688716`](https://github.com/rickstaa/stable-learning-control/commit/4688716d725a1aa10005a6609c1712f3cc327f86)
*   :memo: Updates the README.md formatting [`43a96a7`](https://github.com/rickstaa/stable-learning-control/commit/43a96a76102e2486d686bb5b6b152e994f693830)

## [v0.1.0](https://github.com/rickstaa/stable-learning-control/compare/v0.0.1...v0.1.0) - 2020-12-22

### Merged

*   :arrow_up: Update dependency tensorflow to v2 [`#53`](https://github.com/rickstaa/stable-learning-control/pull/53)
*   Update dependency sphinx-rtd-theme to v0.5.0 [`#49`](https://github.com/rickstaa/stable-learning-control/pull/49)
*   :arrow_up: Update actions/upload-artifact action to v2 [`#50`](https://github.com/rickstaa/stable-learning-control/pull/50)
*   :arrow_up: Update dependency sphinx to v1.8.5 [`#48`](https://github.com/rickstaa/stable-learning-control/pull/48)
*   :arrow_up: Update dependency seaborn to v0.11.0 [`#47`](https://github.com/rickstaa/stable-learning-control/pull/47)
*   :arrow_up: Update dependency matplotlib to v3.3.3 [`#46`](https://github.com/rickstaa/stable-learning-control/pull/46)
*   :arrow_up: Update dependency gym to v0.18.0 [`#45`](https://github.com/rickstaa/stable-learning-control/pull/45)
*   :twisted_rightwards_arrows: Update dependency cloudpickle to ~=1.6.0 [`#43`](https://github.com/rickstaa/stable-learning-control/pull/43)
*   Pin dependencies [`#42`](https://github.com/rickstaa/stable-learning-control/pull/42)
*   :construction_worker: Add renovate.json [`#41`](https://github.com/rickstaa/stable-learning-control/pull/41)
*   :arrow_up: Update gym requirement from ~=0.15.3 to ~=0.18.0 [`#40`](https://github.com/rickstaa/stable-learning-control/pull/40)
*   :bug: Fixes CI config bug [`#29`](https://github.com/rickstaa/stable-learning-control/pull/29)
*   :twisted_rightwards_arrows: Adds wrong code to check the CI pipeline [`#28`](https://github.com/rickstaa/stable-learning-control/pull/28)
*   :twisted_rightwards_arrows: Test ci [`#27`](https://github.com/rickstaa/stable-learning-control/pull/27)
*   :sparkles: Adds gym environment registration [`#13`](https://github.com/rickstaa/stable-learning-control/pull/13)
*   :truck: Changes simzoo folder structure and add setup.py [`#11`](https://github.com/rickstaa/stable-learning-control/pull/11)

### Commits

*   :memo: Updates documentation [`5949d97`](https://github.com/rickstaa/stable-learning-control/commit/5949d9751ca18ca195a0cbbe13fb23e5cd550755)
*   :memo: Updates docs structure [`6bffce6`](https://github.com/rickstaa/stable-learning-control/commit/6bffce6b2464b59ef6569cbb87b97f5eb34af99f)
*   :memo: Adds the documentation [`96c1417`](https://github.com/rickstaa/stable-learning-control/commit/96c1417b24a0eff2a7fd4bca4d47ab6f43d5dac1)
*   :memo: Updates documentation and add documentation CI action [`ed40e0e`](https://github.com/rickstaa/stable-learning-control/commit/ed40e0eea76d2eee9541663dfb9ae83edd4dda8c)
*   :bug: Fixes syntax errors [`fa0d646`](https://github.com/rickstaa/stable-learning-control/commit/fa0d6467bbb089ce7dfc44c9ce4362d0fa965f22)
*   :fire: Removes some unused files and adds some Useful scripts" [`52958b4`](https://github.com/rickstaa/stable-learning-control/commit/52958b4680f7945da1ab0a4c9e2ec1d367fdb871)
*   :bug: Fixes some bugs in the LAC algorithm and updates the documentation [`b73a99b`](https://github.com/rickstaa/stable-learning-control/commit/b73a99bcc7c83c1c6b14dae0941b17f101ff88db)
*   :memo: Updates errors in the documentation. [`d2ac20c`](https://github.com/rickstaa/stable-learning-control/commit/d2ac20c3b205874221ced23e075ab4e31341431d)
*   :art: Improves code syntax [`bcc676c`](https://github.com/rickstaa/stable-learning-control/commit/bcc676cfa3d97efd70b861252a8c222afd340d77)
*   :construction_worker: Adds linux/mac/windows ci checks [`159d8d9`](https://github.com/rickstaa/stable-learning-control/commit/159d8d97dd05783b230197db5e684f80f133bdb8)
*   :green_heart: Perform some CI tests [`bd10542`](https://github.com/rickstaa/stable-learning-control/commit/bd1054296ff23471602b2ac73fd99cffc34c672b)
*   :bug: Fixes simzoo module import errors [`0c172b0`](https://github.com/rickstaa/stable-learning-control/commit/0c172b096e66e3350cb09285c7a306c2167b1926)
*   :memo: Updates README.md [`e81a648`](https://github.com/rickstaa/stable-learning-control/commit/e81a6488e183696fafd349d613f45c7b357f6557)
*   :construction_worker: Make ci spellcheck less strict [`8deebb2`](https://github.com/rickstaa/stable-learning-control/commit/8deebb2876f71f7187faf0a1b523c28d4726ff57)
*   :construction_worker: Fixes github pytest action [`40492b5`](https://github.com/rickstaa/stable-learning-control/commit/40492b583318c849cb298e565534d5c425602f2c)
*   :bug: Fixes github actions cache step [`e57ea50`](https://github.com/rickstaa/stable-learning-control/commit/e57ea50d18e167b158142792db5e3db9b354b22b)
*   :construction_worker: Updates CI action python version [`a29e99e`](https://github.com/rickstaa/stable-learning-control/commit/a29e99ec05090b9674343f0ec008ed9639f85402)
*   :memo: Updates control readme.md [`2a966b5`](https://github.com/rickstaa/stable-learning-control/commit/2a966b55e18cd5e08ac59fc87ff4ffd276a7f22c)
*   :memo: Updates main README.md [`1b8634e`](https://github.com/rickstaa/stable-learning-control/commit/1b8634ecf1bf58015da8d95448c40e51367b27a5)
*   :construction: Adds dummy function to the tf2 lac template [`574f874`](https://github.com/rickstaa/stable-learning-control/commit/574f874bb6328d5d882a4f8179f2faac9e28c342)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`a5aeedd`](https://github.com/rickstaa/stable-learning-control/commit/a5aeedd8be2a45a2ef77f8089f0c9a622f2a3938)
*   :fire: Removes redundant folders [`97cf08c`](https://github.com/rickstaa/stable-learning-control/commit/97cf08c3e625f4241478b47568f3cbbbc2708cfe)
*   :memo: Cleans up documentation [`9376e27`](https://github.com/rickstaa/stable-learning-control/commit/9376e2780d0489dcd385f99acfa227d335503004)
*   :green_heart: Updates gh-actions [`1ed50ba`](https://github.com/rickstaa/stable-learning-control/commit/1ed50ba98b285b24cb6bf8b77a5ed4d7303a2a1a)
*   :bug: Disables broken CI script [`d2f3dc4`](https://github.com/rickstaa/stable-learning-control/commit/d2f3dc449f012ba6881432f291897c528a22408f)
*   :construction_worker: Re-adds docs CI action [`ae324ed`](https://github.com/rickstaa/stable-learning-control/commit/ae324edf777aaf7c034c85192c2624c552782502)
*   :green_heart: Fixes bug in the docs ci action [`8426c92`](https://github.com/rickstaa/stable-learning-control/commit/8426c922de47e036a12d9e0d35e6f4a118c25ea2)
*   :construction_worker: Adds docs build check step [`6afd063`](https://github.com/rickstaa/stable-learning-control/commit/6afd063d7a1b1dd56daf6b83bf1657cdc802c996)
*   :green_heart: Fixes ci build [`90d6f74`](https://github.com/rickstaa/stable-learning-control/commit/90d6f74ff3fa6845142dfa1e9083bf74825932be)
*   :memo: Adds os CI badges to the readme [`c0abc05`](https://github.com/rickstaa/stable-learning-control/commit/c0abc054bad2056f22d3867331dde8f85c62ab55)
*   :green_heart: Updates test gh-action [`6497f23`](https://github.com/rickstaa/stable-learning-control/commit/6497f23584a6d66e1790791b2df0a0690a344430)
*   :bookmark: Bump version: 0.0.1 → 0.1.0 [`5bf3396`](https://github.com/rickstaa/stable-learning-control/commit/5bf3396a231f86de64736f47fcb93885f08e8823)
*   :memo: Updates python version support badge [`4a94070`](https://github.com/rickstaa/stable-learning-control/commit/4a94070d5a0a41caef0a297820f4ce4aabbd54dd)
*   :green_heart: Updates gh-action tests [`8a8f41b`](https://github.com/rickstaa/stable-learning-control/commit/8a8f41bbf8826bf79d7752c3c5735f2d334898d0)
*   :green_heart: Another CI docs release fix [`a7ba90f`](https://github.com/rickstaa/stable-learning-control/commit/a7ba90f23b2687f7a70402e50bb99836059bcd6d)
*   :construction_worker: Fixes github actions syntax error [`6bf75dc`](https://github.com/rickstaa/stable-learning-control/commit/6bf75dc3d4ea044205144fdca376d6d78849ca5d)
*   :construction_worker: Fixes flake8 action repository url [`5dd9c0c`](https://github.com/rickstaa/stable-learning-control/commit/5dd9c0c36caac596c7e11977f5695c1e3fbf777f)
*   :bug: Fixes mlc github action bug [`718a276`](https://github.com/rickstaa/stable-learning-control/commit/718a276e0ac6d5ca991f3cfbd46c1711fe3bcf47)
*   :wrench: Updates bumpversion config [`21d3010`](https://github.com/rickstaa/stable-learning-control/commit/21d3010ea5f40b622934c3da635fb168263d275a)
*   :green_hearth: Adds different tag recognition github action [`e99e4f2`](https://github.com/rickstaa/stable-learning-control/commit/e99e4f219d0571e7733bb675cf0e7c593abbc776)
*   :green_heart: Updates gh-action tests [`5014392`](https://github.com/rickstaa/stable-learning-control/commit/50143925b9993795f1aeb4e8b2e97e6069c67bb3)
*   :green_heart: Adds documentation artifacts [`0444a29`](https://github.com/rickstaa/stable-learning-control/commit/0444a291eef627f46d8a22622719fa247035bd9d)
*   :construction_worker: Change python cache file [`721dad5`](https://github.com/rickstaa/stable-learning-control/commit/721dad5eb552da956b7386508bee450672546dc8)
*   :memo: Updates documentation todos [`7ad1116`](https://github.com/rickstaa/stable-learning-control/commit/7ad1116e946a53800b112831a2f05defd21b37de)
*   :bug: Fixes github actions pip upgrade step [`753b53c`](https://github.com/rickstaa/stable-learning-control/commit/753b53cc473d73a551c0f21b5e3119361376b73b)
*   :green_heart: Changes the gh action tests [`ee8f49e`](https://github.com/rickstaa/stable-learning-control/commit/ee8f49e0d3afe4a088f2ec5bf604cf4dfeb3383a)
*   :green_heart: Adds some gh action tests [`92353e5`](https://github.com/rickstaa/stable-learning-control/commit/92353e5ee7c2669c38d4d7fef4741c6056c4a8bf)
*   :construction_worker: Force github actions to rebuild pip cache [`0449167`](https://github.com/rickstaa/stable-learning-control/commit/04491671f87f45d3a6e865c0a7590c7ae1df7df2)
*   :wrench: Updates bumpversion config [`8d6c4d0`](https://github.com/rickstaa/stable-learning-control/commit/8d6c4d094e41f48a266d5cb3de11ac3793948c1f)
*   :green_heart: Updates gh-actions tests [`cb2e4de`](https://github.com/rickstaa/stable-learning-control/commit/cb2e4de6f418ab622a196592f5f73dc6ff91b3a5)
*   :green_heart: Updates gh action tests [`d8b454a`](https://github.com/rickstaa/stable-learning-control/commit/d8b454a66c99e17dc4a05921f61a0fb7304992e0)
*   :green_heart: Removes redundant gh action step [`5ffb682`](https://github.com/rickstaa/stable-learning-control/commit/5ffb682d0d78314a86a09f6b0f17ffa95f74ca44)
*   :green_heart: Fixes bug in the docs release github action [`8703507`](https://github.com/rickstaa/stable-learning-control/commit/870350776c552702fe86f112fca0bc309aaebd62)
*   :bug: Fixes pytest github action command [`768be0a`](https://github.com/rickstaa/stable-learning-control/commit/768be0aa5981f199fefb0af7eb87993380fc63bd)
*   :construction_worker: Updates github actions checkout token [`30f4ab6`](https://github.com/rickstaa/stable-learning-control/commit/30f4ab65ce11a48c5e5467f02e927a4333ac5427)
*   :sparkles: Re-adds simzoo submodule [`8784a2a`](https://github.com/rickstaa/stable-learning-control/commit/8784a2ab2a64b652d744fe7dc4ab329a73685e08)
*   :fire: Removes redundant submodule [`ab868cd`](https://github.com/rickstaa/stable-learning-control/commit/ab868cd4571c835b97c14716e9e7931b4c9eeadd)
*   :green_heart: Updates gh-actions tests [`01e1643`](https://github.com/rickstaa/stable-learning-control/commit/01e1643135d5e64669621dd2d169ed98231cc9d9)
*   :green_heart: Updates gh-actions tests [`88da879`](https://github.com/rickstaa/stable-learning-control/commit/88da8791beda88bb255565572a8355abcdfaf4a4)
*   :green_heart: Fixes docs ci bug [`c1e7b69`](https://github.com/rickstaa/stable-learning-control/commit/c1e7b69af5b95682860542743bb2ffca2a995d16)
*   :green_heart: Updates docs ci action to create orphan branch [`c8550ee`](https://github.com/rickstaa/stable-learning-control/commit/c8550ee4101b7b2f26197772269da272ff90cac6)
*   :memo: Updates documentation link in the README.md [`db90671`](https://github.com/rickstaa/stable-learning-control/commit/db906712c95ceebfcd265567c41f1fa326291599)
*   :art: :wrench: Updates bumpversion config [`aa404ea`](https://github.com/rickstaa/stable-learning-control/commit/aa404eac2adfef613aac14042c5445dd3fc1b7c2)
*   :bookmark: Updates version [`3ce4702`](https://github.com/rickstaa/stable-learning-control/commit/3ce47025e1d2ddbe3cce390608a2caa79059fc02)
*   :green_heart: Updates gh-actions tests [`daff84b`](https://github.com/rickstaa/stable-learning-control/commit/daff84b40dea4c5c13ad8817112c66d6c35273ea)
*   :green_heart: Fixes docs ci build [`3657569`](https://github.com/rickstaa/stable-learning-control/commit/36575697c177653a844428c5613cc0b0a73bfc62)
*   :green_heart: Updates new docs github action [`48856ec`](https://github.com/rickstaa/stable-learning-control/commit/48856ec91980e4b7908baeafb26b8e72a6d15099)
*   :green_heart: Fixes syntax error in github action [`21776e7`](https://github.com/rickstaa/stable-learning-control/commit/21776e7cacf3ca49033526d11bf0d6b751e65f1f)
*   :construction_worker: Updates CI config files [`45ebf3a`](https://github.com/rickstaa/stable-learning-control/commit/45ebf3a5e8f046a6be073b458cfb097389968f2b)
*   :wrench: Updates sphinx config file [`ed4a5c8`](https://github.com/rickstaa/stable-learning-control/commit/ed4a5c87b0d4d0adc57cde01e8cee352ecee4566)
*   :construction_worker: Fixes flake8 fail on error problem [`baa3332`](https://github.com/rickstaa/stable-learning-control/commit/baa333220d997774db465844956cbe01fd4e62d4)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`3347c98`](https://github.com/rickstaa/stable-learning-control/commit/3347c9861048c68f356d0a9019b88cb4c9895256)
*   :construction_worker: Adds github token to ci action [`50f471d`](https://github.com/rickstaa/stable-learning-control/commit/50f471de08cd696387c63da8df47296a8aa9a230)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`add11ee`](https://github.com/rickstaa/stable-learning-control/commit/add11ee210be0ae1953d2b2967abea3b864e6b41)
*   :bug: Fixes bug in .gitmodules file [`5493f2f`](https://github.com/rickstaa/stable-learning-control/commit/5493f2f0d69729a96e0ca26f0aa6fb4fa8998622)
*   :wrench: Updates gitmodules file to accept relative urls [`4ad5624`](https://github.com/rickstaa/stable-learning-control/commit/4ad5624fd89c5b50a05b32d96860640ff51f34d2)
*   :green_heart: Forces pip cache to rebuild [`d798cb9`](https://github.com/rickstaa/stable-learning-control/commit/d798cb9fd7003f36e400ee412bbcec1423de3b52)
*   :green_heart: Changes docs ci commit message [`81cd922`](https://github.com/rickstaa/stable-learning-control/commit/81cd9229e8167826229c6557da94935e62fd8ccb)
*   :wrench: Adds .nojekyll file to documentation build step [`b5c50c6`](https://github.com/rickstaa/stable-learning-control/commit/b5c50c6d5b76a22c7f0c1131988e02ecbfa595ab)
*   :twisted_rightwards_arrows: Merge branch 'renovate/cloudpickle-1.x' into main [`59af010`](https://github.com/rickstaa/stable-learning-control/commit/59af0106da22af2f975f1b4b07243ce632a5c19a)
*   :twisted_rightwards_arrows: Merge branch 'dependabot/pip/docs/tensorflow-2.4.0' into main [`56548ca`](https://github.com/rickstaa/stable-learning-control/commit/56548ca544966b22958dabdb07f537d43c30a8a8)
*   :twisted_rightwards_arrows: Merge branch 'main' into dependabot/pip/docs/tensorflow-2.4.0 [`df9aa73`](https://github.com/rickstaa/stable-learning-control/commit/df9aa7397a362d456d0db93cb5f6967457b104fe)
*   :twisted_rightwards_arrows: Merge branch 'main' into renovate/cloudpickle-1.x [`6de29ab`](https://github.com/rickstaa/stable-learning-control/commit/6de29ab3a2de1102b52825d7e54a6223ce743e8c)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`a27a0f3`](https://github.com/rickstaa/stable-learning-control/commit/a27a0f3eb0f154663cb612723d6fee414741b2d2)
*   :wrench: Updates setup.py and documentation [`ecba2c7`](https://github.com/rickstaa/stable-learning-control/commit/ecba2c786b0cb43d181c01896a19aa23082e79a3)
*   :green_heart: Updates github actions [`3a85ad4`](https://github.com/rickstaa/stable-learning-control/commit/3a85ad4c2c728d35d0d34938c66320e34b70613d)
*   :green_heart: Updates github actions [`a6a7c28`](https://github.com/rickstaa/stable-learning-control/commit/a6a7c28e70749e03134693e3d21b560589e8e798)
*   :wrench: Migrate to pyproject.toml (PEP518/517) [`e564b2e`](https://github.com/rickstaa/stable-learning-control/commit/e564b2ea06ada16e5988d0ccbe5ed56bee0ad7eb)
*   :bug: Fixes CI config bugs [`d9da3fd`](https://github.com/rickstaa/stable-learning-control/commit/d9da3fd0ba7b9b61659b7ebfe97b201313f4e97e)
*   :recycle: Cleanup CI config file [`538ae20`](https://github.com/rickstaa/stable-learning-control/commit/538ae2086ae1b616b5be0523a2f7abc34c96700d)
*   :bug: Fixes a bug that was introduced due to e564b2ea06 [`a03c2d0`](https://github.com/rickstaa/stable-learning-control/commit/a03c2d0bab9a637f954ba271a925a3e92c8fc56c)
*   :fire: Removes redundant action files [`4b499ee`](https://github.com/rickstaa/stable-learning-control/commit/4b499eef35f0175954ddfef94ce5c866d4f35f56)
*   :fire: Removes redundant files and updates dev reqs [`97f1f6d`](https://github.com/rickstaa/stable-learning-control/commit/97f1f6d5768b2c0e88e87b51e81fb6b49bbb8362)
*   :bug: Updates pytests [`943badf`](https://github.com/rickstaa/stable-learning-control/commit/943badfeeeb134fa9db9d29696203f9e309e0625)
*   :wastebasket: Removes depricated code and updates CI config [`b2f4187`](https://github.com/rickstaa/stable-learning-control/commit/b2f4187fc16efe0cfcddf0994dcef2c43d996f41)
*   :wrench: Add security config [`f66d4c5`](https://github.com/rickstaa/stable-learning-control/commit/f66d4c58fe305cfa117d2bb095dd7a97560c2bb6)
*   :fire: Removes redundant requirements file [`d5a5436`](https://github.com/rickstaa/stable-learning-control/commit/d5a543626d9c21742e965e5e5154433d0746a08b)
*   :bug: Removes redundant CI config if statements [`d3acd62`](https://github.com/rickstaa/stable-learning-control/commit/d3acd62f3c2683736b25d9061a37dd3ad06a1847)
*   :wrench: Updates CI config [`aeca3cf`](https://github.com/rickstaa/stable-learning-control/commit/aeca3cf8f5a2a57712c4094d14fcd33670fe56fa)
*   :bug: Fixes CI config [`94335d8`](https://github.com/rickstaa/stable-learning-control/commit/94335d81adb5fca1bc1bd53883fd1dfe8f5ca824)
*   :memo: Updates documentation and removes redundant files [`b7436f3`](https://github.com/rickstaa/stable-learning-control/commit/b7436f3d1811261e4032b5d6693ebcf919c06c65)
*   :wrench: Adds dependabot configuration file [`62098ca`](https://github.com/rickstaa/stable-learning-control/commit/62098caecaa62d5b575ca19f9f0ec390cebfd66d)
*   :recycle: Clean up CI script [`1bab689`](https://github.com/rickstaa/stable-learning-control/commit/1bab68975e3d43c2e4cf5d9effcbd085373cf487)
*   :fire: Removes unused files and updates .gitignore [`53415a3`](https://github.com/rickstaa/stable-learning-control/commit/53415a3624b955bb1d6ed1a391a6182d1ad84bc9)
*   :bug: Adds missing system dependencies to CI [`5b981e5`](https://github.com/rickstaa/stable-learning-control/commit/5b981e5db4f779263d18e4bb74c7a9040b4bf880)
*   Update dependency cloudpickle to v1.6.0 [`2fb8c5d`](https://github.com/rickstaa/stable-learning-control/commit/2fb8c5d9dfc21aaee96a4c5049f4aca55323f50c)
*   :arrow_up: Bump tensorflow from 1.15.4 to 2.4.0 in /docs [`c7cafff`](https://github.com/rickstaa/stable-learning-control/commit/c7caffff1b9161e975e102fe726ccabe8eaf77f9)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`23fb67e`](https://github.com/rickstaa/stable-learning-control/commit/23fb67e25682f157978690831f5850dc9b4f282d)
*   :construction_worker: Updates remark ci action script [`11e22e1`](https://github.com/rickstaa/stable-learning-control/commit/11e22e13f38bbf931507d9cd770a15b8dc9c1190)
*   :twisted_rightwards_arrows: Merge branch 'add_lac_algorithm' into main [`81b88cd`](https://github.com/rickstaa/stable-learning-control/commit/81b88cdba72aa4e4d97f9ed0319fa798ca42049c)
*   :alien: Updates submodules [`92e6575`](https://github.com/rickstaa/stable-learning-control/commit/92e65757d9bfcda8d760c1bedf96717cbf252707)
*   :bug: Fix CI config syntax error [`b0a67ea`](https://github.com/rickstaa/stable-learning-control/commit/b0a67eaef8ffd0621d5d40cb41e8cb7f0bc5e635)
*   :bug: Always run cache CI [`49174cb`](https://github.com/rickstaa/stable-learning-control/commit/49174cbcb20daa0894ba80547b05c9ba3fa085ed)
*   :bug: Set CI failfast to false [`8445143`](https://github.com/rickstaa/stable-learning-control/commit/8445143b5e1dcd9b9f674fca0c5c613b3570024a)
*   :wrench: Updates CI config file [`7bc1c73`](https://github.com/rickstaa/stable-learning-control/commit/7bc1c73b325efad232089da27964ab8cc8880e36)
*   :bug: Fixes CI python cache id [`6553190`](https://github.com/rickstaa/stable-learning-control/commit/65531906a53415fb6947336977321776c7c60422)
*   :bug: Fixes bug in github actions [`b239fc3`](https://github.com/rickstaa/stable-learning-control/commit/b239fc310869816bddc077c0b2274f6a9f40e772)
*   :bug: Fixes bug in github actions [`533e873`](https://github.com/rickstaa/stable-learning-control/commit/533e873a9c680b1f57b49b36b959bb7b3f28c792)
*   :wrench: Updates dependabot config file [`db109ec`](https://github.com/rickstaa/stable-learning-control/commit/db109ecb51b19a6c0292e576c1bff8923553a39e)
*   :alien: Updates simzoo submodule [`7ecaca1`](https://github.com/rickstaa/stable-learning-control/commit/7ecaca1bd53fac5460059c9c3b83aaf0453e48ee)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`d8f8450`](https://github.com/rickstaa/stable-learning-control/commit/d8f8450f6fff6026dfba8f15d6a9a16921b3340b)
*   :sparkles: Adds the LAC algorithm [`cd51914`](https://github.com/rickstaa/stable-learning-control/commit/cd51914e0d27b0464511037da945191e6adebbbd)
*   :sparkles: Fixes wrong learning rate decay in the SAC algorithm [`62cf02c`](https://github.com/rickstaa/stable-learning-control/commit/62cf02ca1f5ff3acea6267dd72131306874c57a5)
*   :bug: Fixes some small bugs in the LAC algorithm [`0591e07`](https://github.com/rickstaa/stable-learning-control/commit/0591e07c726f454a129dbd66fb3136cb1130e88a)
*   :sparkles: Adds SAC Algorithm [`3cb1ea4`](https://github.com/rickstaa/stable-learning-control/commit/3cb1ea4dd292608a0c05daa8272ad8dde655b9a7)
*   :sparkles: Adds working SAC implementation [`4cac2cf`](https://github.com/rickstaa/stable-learning-control/commit/4cac2cfca7bbab987b46881d3491c8641a502090)
*   :fire: Removes unused code and files [`7a9889d`](https://github.com/rickstaa/stable-learning-control/commit/7a9889d739e04a8bff0f671c27cf3592efe2f1ee)
*   :pencil: Adds documentation template [`fbafec2`](https://github.com/rickstaa/stable-learning-control/commit/fbafec25cce474a2694317f18afbe3a3409e71f8)
*   :pencil: Updates README.md and cleans up LAC and SAC code [`e76c0e8`](https://github.com/rickstaa/stable-learning-control/commit/e76c0e8e858e9e02d0c8606226d1274c4e31dee9)
*   :bug: Test pip cache [`85f1380`](https://github.com/rickstaa/stable-learning-control/commit/85f1380a99d0498ad14a64e3d47400695e7aa2e0)
*   :sparkles: Changes the SAC learning rate decay from step to episode based [`08fea9a`](https://github.com/rickstaa/stable-learning-control/commit/08fea9a83addf2c5987367e6528ae1cf090690af)
*   :wrench: Updates CI config [`6289ae3`](https://github.com/rickstaa/stable-learning-control/commit/6289ae392c732f7443387d14c9b71248c4e9c326)
*   :art: Improves code readability and pulls submodules [`4efa66b`](https://github.com/rickstaa/stable-learning-control/commit/4efa66b8df1c8816c856d95474d6899f49f6a17c)
*   :sparkles: Adds multiple python versions tests to CI [`616e592`](https://github.com/rickstaa/stable-learning-control/commit/616e592a07140129a0f9624de3436d87f88b20f1)
*   :bug: Fixes increasing lambda error [`ede5d6b`](https://github.com/rickstaa/stable-learning-control/commit/ede5d6bf3beeb04874621d1888a6378579d2ff14)
*   :art: Improves code formatting [`5db397a`](https://github.com/rickstaa/stable-learning-control/commit/5db397ae0f281ff1171091a3a80733e752042873)
*   :pencil: Updates README, removes unused files and pulls submodules [`9766020`](https://github.com/rickstaa/stable-learning-control/commit/9766020c486e4691ffae1882ad7514e1f7675905)
*   :bug: Fixes CI cache bug [`afb146d`](https://github.com/rickstaa/stable-learning-control/commit/afb146d6372a3a68410deddaf43a67e18126a268)
*   :bug: Changes flak8 action [`e4b549a`](https://github.com/rickstaa/stable-learning-control/commit/e4b549a2be8a8dda5b96097dbfe5dac828a1b460)
*   :bug: Changes flake8 CI action provider [`1237885`](https://github.com/rickstaa/stable-learning-control/commit/1237885d402a3c6b2f6fb3aebea0fe9c684e88d7)
*   :bug: Fixes the algorithm run scripts to have the right arguments [`55ef144`](https://github.com/rickstaa/stable-learning-control/commit/55ef14479f9e1ddb30c170d97be05ce68720330b)
*   :bug: Fixes flake 8 CI command error [`4362067`](https://github.com/rickstaa/stable-learning-control/commit/4362067712d04821abd1f153e97ae16733016343)
*   :bug: Changes flak8 anotator CI [`945083d`](https://github.com/rickstaa/stable-learning-control/commit/945083d754598d70c2547bdf177fdb13af498525)
*   :sparkles: Updates simzoo submodule [`4a4b73b`](https://github.com/rickstaa/stable-learning-control/commit/4a4b73b6ade6f64ba1d00eeb8995151a3bd1a02e)
*   :bug: Updates CI script formatting [`c06fc75`](https://github.com/rickstaa/stable-learning-control/commit/c06fc7588292c33f3c7361c7c9403fb6ec0a2a7c)
*   :bug: Fixes pytest artifacts upload bug CI [`5b73136`](https://github.com/rickstaa/stable-learning-control/commit/5b73136eaed361e41b1cd3d9f0fb0566c41194f1)
*   :bug: Fixes CI format bug [`25671b6`](https://github.com/rickstaa/stable-learning-control/commit/25671b6e4d1c9b15d0da4c6aa590ed8a8c4960e7)
*   :bug: Fixes CI python cache problem [`0ac5fa9`](https://github.com/rickstaa/stable-learning-control/commit/0ac5fa9a2174257b412240f03e099c87b4ad96dc)
*   :bug: Fix CI syntax error [`dd8356b`](https://github.com/rickstaa/stable-learning-control/commit/dd8356b522ee9ec1f1038c8ccc06140412ede989)
*   :bug: Fixes remark link CI config [`ff9005a`](https://github.com/rickstaa/stable-learning-control/commit/ff9005a1a47848abe82eba6dc9ec59eddea70cbe)
*   :bug; Fix CI cache operation [`51c126e`](https://github.com/rickstaa/stable-learning-control/commit/51c126ed33c2dacbfa61333a7ce114e593a08d02)
*   :bug: Updates markdown CI config [`7cecd82`](https://github.com/rickstaa/stable-learning-control/commit/7cecd820b502c91215ca2d4a7085caa2dbd95adb)
*   :bug: Fixes pylint bug [`540bd53`](https://github.com/rickstaa/stable-learning-control/commit/540bd53b2ae929b784a6d1033f88291589285050)
*   :bug: Fixes CI format bug [`5018c9e`](https://github.com/rickstaa/stable-learning-control/commit/5018c9e0a7d1dae864bf2dc0e746245aca18582f)
*   :bug: Fix CI config syntax bug [`bd760f0`](https://github.com/rickstaa/stable-learning-control/commit/bd760f056da9568772a53a55ce12e7f34534fa0c)
*   :bug: Fixes dev python dependencies [`3733f18`](https://github.com/rickstaa/stable-learning-control/commit/3733f18ef13c04980acc5f2c4ae0793c5929a60b)
*   :bug; Fix CI cache operation [`dcffef6`](https://github.com/rickstaa/stable-learning-control/commit/dcffef632a7e89fd503cf8a423121638b4ec9156)
*   :bug: Fix CI syntax error [`b322543`](https://github.com/rickstaa/stable-learning-control/commit/b32254366b3ad9131a7bf47e56f683f014924852)
*   :bug: Fixes CI black files bug [`add3364`](https://github.com/rickstaa/stable-learning-control/commit/add3364821c2a29e1710690f6848f2d3bd8e949c)
*   :twisted_rightwards_arrows: Merges branch add_lac_algorithm into master [`7457680`](https://github.com/rickstaa/stable-learning-control/commit/74576805264f24f86b3ce51027d8750b7bbfd3f5)
*   :sparkles: Updates environments [`829bf94`](https://github.com/rickstaa/stable-learning-control/commit/829bf94f4965e0b52b7116e81d0684597ca7f27c)
*   :sparkles: Updates simzoo submodule [`bfb75ea`](https://github.com/rickstaa/stable-learning-control/commit/bfb75eaf76435f918372d7299847f0dc8225d727)

## [v0.0.1](https://github.com/rickstaa/stable-learning-control/compare/v0...v0.0.1) - 2020-08-14

## [v0](https://github.com/rickstaa/stable-learning-control/compare/v0.0.0...v0) - 2021-03-25

### Merged

*   :bug: Fixes --exp_cfg run bug [`#118`](https://github.com/rickstaa/stable-learning-control/pull/118)
*   :green_heart: Makes sure OS actions don't run on doc change [`#117`](https://github.com/rickstaa/stable-learning-control/pull/117)
*   :memo: Updates documentation header [`#116`](https://github.com/rickstaa/stable-learning-control/pull/116)
*   :green_heart: Fixes OS gh actions [`#115`](https://github.com/rickstaa/stable-learning-control/pull/115)
*   :sparkles: Adds robustness eval utility [`#114`](https://github.com/rickstaa/stable-learning-control/pull/114)
*   :memo: Update the documentation and cleans up the code [`#113`](https://github.com/rickstaa/stable-learning-control/pull/113)
*   :bug: Makes the test_policy utility compatible with tf2 [`#112`](https://github.com/rickstaa/stable-learning-control/pull/112)
*   :sparkles: Adds Torch GPU support [`#111`](https://github.com/rickstaa/stable-learning-control/pull/111)
*   :sparkles: Adds TF tensorboard support [`#110`](https://github.com/rickstaa/stable-learning-control/pull/110)
*   :bug: Fixes plotter [`#107`](https://github.com/rickstaa/stable-learning-control/pull/107)
*   :art: Format Python code with psf/black [`#106`](https://github.com/rickstaa/stable-learning-control/pull/106)
*   :bug: Fixes small learning rate lower bound bug [`#105`](https://github.com/rickstaa/stable-learning-control/pull/105)
*   :art: Format Python code with psf/black [`#104`](https://github.com/rickstaa/stable-learning-control/pull/104)
*   :arrow_up: Update dependency sphinx to v3.5.1 [`#101`](https://github.com/rickstaa/stable-learning-control/pull/101)
*   :sparkles: Add tf2 lac version [`#102`](https://github.com/rickstaa/stable-learning-control/pull/102)
*   :arrow_up: Update dependency torch to v1.7.1 [`#58`](https://github.com/rickstaa/stable-learning-control/pull/58)
*   :sparkles: Adds torch lac algorithm [`#100`](https://github.com/rickstaa/stable-learning-control/pull/100)
*   :art: Format Python code with psf/black [`#95`](https://github.com/rickstaa/stable-learning-control/pull/95)
*   :art: Format Python code with psf/black [`#89`](https://github.com/rickstaa/stable-learning-control/pull/89)
*   :rotating_light: fix flake8 errors [`#88`](https://github.com/rickstaa/stable-learning-control/pull/88)
*   :art: Cleanup code [`#87`](https://github.com/rickstaa/stable-learning-control/pull/87)
*   :art: Format Python code with psf/black [`#86`](https://github.com/rickstaa/stable-learning-control/pull/86)
*   :art: Format Python code with psf/black [`#85`](https://github.com/rickstaa/stable-learning-control/pull/85)
*   :arrow_up: Update dependency sphinx to v3 [`#51`](https://github.com/rickstaa/stable-learning-control/pull/51)
*   :arrow_up: Update dependency sphinx-autobuild to v2020 [`#52`](https://github.com/rickstaa/stable-learning-control/pull/52)
*   :arrow_up: Update dependency sphinx-rtd-theme to v0.5.1 [`#65`](https://github.com/rickstaa/stable-learning-control/pull/65)
*   :pushpin: Pin dependency auto-changelog to 2.2.1 [`#84`](https://github.com/rickstaa/stable-learning-control/pull/84)
*   :recycle: Cleanup code [`#83`](https://github.com/rickstaa/stable-learning-control/pull/83)
*   :green_heart: Adds autotag on labeled pull request gh-action [`#82`](https://github.com/rickstaa/stable-learning-control/pull/82)
*   :green_heart: Adds auto-changelog action [`#81`](https://github.com/rickstaa/stable-learning-control/pull/81)
*   :white_check_mark: Test version release action [`#79`](https://github.com/rickstaa/stable-learning-control/pull/79)
*   :bug: Fixes bug in release gh-action [`#78`](https://github.com/rickstaa/stable-learning-control/pull/78)
*   :green_heart: Ci/gh actions [`#77`](https://github.com/rickstaa/stable-learning-control/pull/77)
*   :green_heart: Seperates the release gh-action into seperate jobs [`#76`](https://github.com/rickstaa/stable-learning-control/pull/76)
*   :green_heart: Fix gh actions [`#75`](https://github.com/rickstaa/stable-learning-control/pull/75)
*   :green_heart: fix gh actions [`#74`](https://github.com/rickstaa/stable-learning-control/pull/74)
*   :green_heart: Updates mac github actions [`#72`](https://github.com/rickstaa/stable-learning-control/pull/72)
*   :art: Format Python code with psf/black [`#70`](https://github.com/rickstaa/stable-learning-control/pull/70)
*   :art: Format Python code with psf/black [`#69`](https://github.com/rickstaa/stable-learning-control/pull/69)
*   :art: Format markdown code with remark-lint [`#68`](https://github.com/rickstaa/stable-learning-control/pull/68)
*   :alien: Updates simzoo submodule [`#62`](https://github.com/rickstaa/stable-learning-control/pull/62)
*   :green_heart: Fixes gh-actions [`#61`](https://github.com/rickstaa/stable-learning-control/pull/61)
*   :arrow_up: Update dependency tensorflow to v2 [`#53`](https://github.com/rickstaa/stable-learning-control/pull/53)
*   Update dependency sphinx-rtd-theme to v0.5.0 [`#49`](https://github.com/rickstaa/stable-learning-control/pull/49)
*   :arrow_up: Update actions/upload-artifact action to v2 [`#50`](https://github.com/rickstaa/stable-learning-control/pull/50)
*   :arrow_up: Update dependency sphinx to v1.8.5 [`#48`](https://github.com/rickstaa/stable-learning-control/pull/48)
*   :arrow_up: Update dependency seaborn to v0.11.0 [`#47`](https://github.com/rickstaa/stable-learning-control/pull/47)
*   :arrow_up: Update dependency matplotlib to v3.3.3 [`#46`](https://github.com/rickstaa/stable-learning-control/pull/46)
*   :arrow_up: Update dependency gym to v0.18.0 [`#45`](https://github.com/rickstaa/stable-learning-control/pull/45)
*   :twisted_rightwards_arrows: Update dependency cloudpickle to ~=1.6.0 [`#43`](https://github.com/rickstaa/stable-learning-control/pull/43)
*   Pin dependencies [`#42`](https://github.com/rickstaa/stable-learning-control/pull/42)
*   :construction_worker: Add renovate.json [`#41`](https://github.com/rickstaa/stable-learning-control/pull/41)
*   :arrow_up: Update gym requirement from ~=0.15.3 to ~=0.18.0 [`#40`](https://github.com/rickstaa/stable-learning-control/pull/40)
*   :bug: Fixes CI config bug [`#29`](https://github.com/rickstaa/stable-learning-control/pull/29)
*   :twisted_rightwards_arrows: Adds wrong code to check the CI pipeline [`#28`](https://github.com/rickstaa/stable-learning-control/pull/28)
*   :twisted_rightwards_arrows: Test ci [`#27`](https://github.com/rickstaa/stable-learning-control/pull/27)
*   :sparkles: Adds gym environment registration [`#13`](https://github.com/rickstaa/stable-learning-control/pull/13)
*   :truck: Changes simzoo folder structure and add setup.py [`#11`](https://github.com/rickstaa/stable-learning-control/pull/11)

### Commits

*   :memo: Updates CHANGELOG.md [`0debf73`](https://github.com/rickstaa/stable-learning-control/commit/0debf73a8ac192ea7c65ed620f34e78ebc6e1f5e)
*   :bookmark: Updates code version to v0.9.6 [`204b575`](https://github.com/rickstaa/stable-learning-control/commit/204b575d7365e929ab296f79a8264da7ae8d9825)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`f9f75f4`](https://github.com/rickstaa/stable-learning-control/commit/f9f75f48ca65701959182776df4026c37c1d48a1)
*   :truck: Rename repository to stable-learning-control [`a5024c9`](https://github.com/rickstaa/stable-learning-control/commit/a5024c936ccdea35d89dc924332107d2c0c851a4)
*   :memo: Updates CHANGELOG.md [`336c15e`](https://github.com/rickstaa/stable-learning-control/commit/336c15e81f3c9bfab2f28601f8cc46075c8f9d46)
*   :bug: Fixes logger checkpoint save bug [`ac4d3c2`](https://github.com/rickstaa/stable-learning-control/commit/ac4d3c2e53ad5b4f0cf4b1568f7585b8cd9f8795)
*   :memo: Updates CHANGELOG.md [`664159b`](https://github.com/rickstaa/stable-learning-control/commit/664159bba6192de000c7c1fe70b4ccd8e40e1ce9)
*   :memo: Updates CHANGELOG.md [`5c442f2`](https://github.com/rickstaa/stable-learning-control/commit/5c442f2cf3dceaa78dca319484c41da6f88772ea)
*   :memo: Updates CHANGELOG.md [`9afc8a5`](https://github.com/rickstaa/stable-learning-control/commit/9afc8a518dc81f0500f78c89c48ae039414b4d86)
*   :memo: Updates documentation [`b2f09be`](https://github.com/rickstaa/stable-learning-control/commit/b2f09befb454e5d1ebf5507d3f73c2ff7a8673aa)
*   :memo: Updates CHANGELOG.md [`2b46dd1`](https://github.com/rickstaa/stable-learning-control/commit/2b46dd1570a5fc823b726711008506aa30d7c772)
*   :memo: Updates CHANGELOG.md [`90d4894`](https://github.com/rickstaa/stable-learning-control/commit/90d4894508123a16444661789b0eaec27c1845c4)
*   :memo: Updates CHANGELOG.md [`6f0893d`](https://github.com/rickstaa/stable-learning-control/commit/6f0893d4b7b0717313843613a17e1e7d134c6410)
*   :wrench: Switched from katex latex engine to imgmath [`a337802`](https://github.com/rickstaa/stable-learning-control/commit/a3378026cbc843ba0043a46625cabfc4e7d45d76)
*   :green_heart: Updates gh-actions so that they also run on main push [`d3279bf`](https://github.com/rickstaa/stable-learning-control/commit/d3279bfcf36de8b87fd14d33be0f399f5bc858a5)
*   :green_heart: Trims down docs release gh-action size [`d3ad44f`](https://github.com/rickstaa/stable-learning-control/commit/d3ad44f17934c80120e468bd0fb605ebd0bc4143)
*   :memo: Adds MPI install instructions to docs [`2f1a210`](https://github.com/rickstaa/stable-learning-control/commit/2f1a210952fd0947863744b734c7ff383dd7682a)
*   :green_heart: Adds texlive-full to docs build gh-action [`d61ed01`](https://github.com/rickstaa/stable-learning-control/commit/d61ed0150dcbf78b23335d96a09c126754178a2e)
*   :bookmark: Updates code version to v0.9.5 [`248e0af`](https://github.com/rickstaa/stable-learning-control/commit/248e0af0e0e6ca48d2cb56e071fd31f03acb151a)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`6c2a0a6`](https://github.com/rickstaa/stable-learning-control/commit/6c2a0a6dd2afbb36e8b0efdcc9a664771226ac10)
*   :building_construction: Fixed simzoo submodule setup [`100aeb2`](https://github.com/rickstaa/stable-learning-control/commit/100aeb255f9b424cc22567d4ff0ca8648f0a1475)
*   :memo: Updates CHANGELOG.md [`0574672`](https://github.com/rickstaa/stable-learning-control/commit/05746727b690f95aeb00e61bc0c40e6f033edfe2)
*   :sparkles: Adds checkpoint load ability to 'test_policy' utility [`59e28fe`](https://github.com/rickstaa/stable-learning-control/commit/59e28fe0ec5857790f6c6049229b601d85cb0281)
*   :green_heart: Add sphinx-latexpdf container to docs release gh-action [`ba7088d`](https://github.com/rickstaa/stable-learning-control/commit/ba7088df6872fff630f872c615584327d3ae1281)
*   :green_heart: Fixes package naming inside the github-actions [`c8f080d`](https://github.com/rickstaa/stable-learning-control/commit/c8f080d967cb2ab85fc3edd8d3eb46f02931670c)
*   :bookmark: Updates code version to v0.9.4 [`6c1e72a`](https://github.com/rickstaa/stable-learning-control/commit/6c1e72ad70ecca4b6f569fb8aca536a2d8415f96)
*   :bookmark: Updates code version to v0.9.2 [`38d537a`](https://github.com/rickstaa/stable-learning-control/commit/38d537a1b18d31a900eb4af491c45f0dd0de4ab8)
*   :bookmark: Updates code version to v0.9.1 [`a4d218f`](https://github.com/rickstaa/stable-learning-control/commit/a4d218ff55d1187c32b69902b89bdf08b31799cf)
*   :bookmark: Updates code version to v0.9.0 [`08204a5`](https://github.com/rickstaa/stable-learning-control/commit/08204a54691ee0ac61de43fc96bc964e7b692e82)
*   :wrench: Updates docs make file [`31217d4`](https://github.com/rickstaa/stable-learning-control/commit/31217d4077f5d27e9c42e24a791eef25c4a9ce06)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`99ff0d5`](https://github.com/rickstaa/stable-learning-control/commit/99ff0d5b6e5bca3b635604094bf823ff56be5e5e)
*   :bookmark: Updates code version to v0.9.3 [`91f5600`](https://github.com/rickstaa/stable-learning-control/commit/91f560080eda0cc1e909b2542d983e8611552739)
*   :bookmark: Updates code version to v0.8.3 [`3864255`](https://github.com/rickstaa/stable-learning-control/commit/386425557239960c939e5784d9f78c438e778b0f)
*   :bookmark: Updates code version to v0.8.1 [`8656bdc`](https://github.com/rickstaa/stable-learning-control/commit/8656bdc5450f951bf9dc66447718797f67c62a5f)
*   :fire: Removes redundant readme's [`76db499`](https://github.com/rickstaa/stable-learning-control/commit/76db49934dd7b774593a71572880445c5447901d)
*   :bulb: Updates code comments [`67ae469`](https://github.com/rickstaa/stable-learning-control/commit/67ae469e409b7a18b614973128c8682f4c78f564)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`711d62b`](https://github.com/rickstaa/stable-learning-control/commit/711d62b4bf29c3beff140dc5088ef3f51243ca15)
*   :green_heart: Fixes apt update bug inside gh-actions [`782f2f9`](https://github.com/rickstaa/stable-learning-control/commit/782f2f9d0765ad60cd2088885be53c34ad8f31e1)
*   :wrench: Changed docs latex engine to katex [`d00f989`](https://github.com/rickstaa/stable-learning-control/commit/d00f98992fb3055a96f73a49a1c6d7197d1c135e)
*   :bug: Fixed corrupt submodules [`0c76d7d`](https://github.com/rickstaa/stable-learning-control/commit/0c76d7d835f78765875480516d14e395ad926c5c)
*   :alien: Updates the simzoo submodule [`c13e770`](https://github.com/rickstaa/stable-learning-control/commit/c13e770c95b84782ce67bc71456057812208f8cf)
*   :wrench: Makes submodules work both with ssh and https [`96b54f0`](https://github.com/rickstaa/stable-learning-control/commit/96b54f017f2ae19fa62ed02e5664ce00f4c5fc9f)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`3fb9256`](https://github.com/rickstaa/stable-learning-control/commit/3fb925603c3df1286c8974b0a2c1e07577703fa0)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-learning-control into main [`9caeccb`](https://github.com/rickstaa/stable-learning-control/commit/9caeccb95b1147af2f7857d572c1463777e157ff)
*   :memo: Adds API documentation [`16ed4f5`](https://github.com/rickstaa/stable-learning-control/commit/16ed4f5bc2366b0700048f7ffd8de9a5e5648b0b)
*   :memo: Adds new --exp-cfg option to the docs [`9684404`](https://github.com/rickstaa/stable-learning-control/commit/96844047143571a111756214c5adae3ba177cd88)
*   :art: Cleans up code and removes redundant files [`ed84b13`](https://github.com/rickstaa/stable-learning-control/commit/ed84b133b7bec2ab3d0abb78e1f88883383e5ae8)
*   :sparkles: Adds torch tensorboard support [`885db1f`](https://github.com/rickstaa/stable-learning-control/commit/885db1f2c6437f514a2b7690379341b2033cdf9b)
*   :memo: Cleans up documentation [`cf97a97`](https://github.com/rickstaa/stable-learning-control/commit/cf97a97d2ac006fe704e2ee9997efebea74dfc19)
*   :sparkles: Adds custom gym environment support [`4d23b1e`](https://github.com/rickstaa/stable-learning-control/commit/4d23b1e9270abdbc2727e56606f7f4743ed2e6c7)
*   :memo: Updates CHANGELOG.md [`e97c34c`](https://github.com/rickstaa/stable-learning-control/commit/e97c34c52482cb65570ff6826d32aad60a995652)
*   :memo: Updates CHANGELOG.md [`fa8875f`](https://github.com/rickstaa/stable-learning-control/commit/fa8875fcc4e23f074655c682c74fd854f940a25f)
*   :memo: Updates CHANGELOG.md [`3dca15c`](https://github.com/rickstaa/stable-learning-control/commit/3dca15c87dafb30b113acc88fc65cb6f29777a15)
*   :memo: Updates CHANGELOG.md [`9a9533f`](https://github.com/rickstaa/stable-learning-control/commit/9a9533f56cdc6bd2e839cf70e6e78ee5ffc7f2f8)
*   :sparkles: Adds --exp-cfg cmd-line argument [`e084ccf`](https://github.com/rickstaa/stable-learning-control/commit/e084ccf0f6bd3ae55762f1816aa061a5352a8a17)
*   :bug: Fixes plotter bug while cleaning up code [`cd8e2a2`](https://github.com/rickstaa/stable-learning-control/commit/cd8e2a291df13341d7ee758bde0e44f524d18031)
*   :wrench: Adds submodule pull check to the setup.py [`e313593`](https://github.com/rickstaa/stable-learning-control/commit/e3135930902dad06db8fb59e6aa9a851088a3e3a)
*   :memo: Updates CHANGELOG.md [`2a3b6bf`](https://github.com/rickstaa/stable-learning-control/commit/2a3b6bf9071f9a8b94e01133feb2bd930f2e64e2)
*   :sparkles: Adds a better method for loading custom gym environments [`f2e7d30`](https://github.com/rickstaa/stable-learning-control/commit/f2e7d309096e7c44a7c7919736411611210d406d)
*   :art: Cleans up code structure [`bc7b485`](https://github.com/rickstaa/stable-learning-control/commit/bc7b485a85883453fa21fb823d4a8e7c478f5f68)
*   :memo: Adds auto pre-release github action [`fa304f3`](https://github.com/rickstaa/stable-learning-control/commit/fa304f3b8455bccc2f41f1b6065529bd6606c385)
*   :memo: Updates CHANGELOG.md [`ae5ab82`](https://github.com/rickstaa/stable-learning-control/commit/ae5ab82a406be8751419d78d2dacee1f26684e6f)
*   :wrench: Updates --exp-cfg example configuration file [`983f9e5`](https://github.com/rickstaa/stable-learning-control/commit/983f9e59c5d480a10a62fdc5a24a5c37d28c66db)
*   :green_heart: Disables gh-actions on direct push to main [`d9c3fdf`](https://github.com/rickstaa/stable-learning-control/commit/d9c3fdf1a774aa757ff50e9e0c1f5001c60445ae)
*   :art: Cleans up plotter code [`2145e17`](https://github.com/rickstaa/stable-learning-control/commit/2145e1715d9c68b493087aa3d94058f35164a74d)
*   :bug: Fixes lr lower bound bug [`dff234e`](https://github.com/rickstaa/stable-learning-control/commit/dff234e34497a307dc25997c243798a4f14bf635)
*   :memo: Updates CHANGELOG.md [`733227a`](https://github.com/rickstaa/stable-learning-control/commit/733227a318b02edabdc0969d47b2267786ccaad8)
*   :bookmark: Updates code version to v0.7.0 [`44b87ec`](https://github.com/rickstaa/stable-learning-control/commit/44b87ec35c385d50713c03a696c5ec381a1f108b)
*   :bookmark: Updates code version to v0.6.0 [`8f816e2`](https://github.com/rickstaa/stable-learning-control/commit/8f816e273c7cdeb1c1446b9ba43c813459a11b7e)
*   :bookmark: Updates code version to v0.4.0 [`da356fe`](https://github.com/rickstaa/stable-learning-control/commit/da356feb7c2fe590a64d9769453d40164cc16085)
*   :memo: Updates CHANGELOG.md [`5da98ef`](https://github.com/rickstaa/stable-learning-control/commit/5da98ef083c89e6d7225b97276333a133ffba10b)
*   :bookmark: Updates documentation version to v0.2.8 [`9971c2d`](https://github.com/rickstaa/stable-learning-control/commit/9971c2d52915a93a7ed4907586644fb6f889cc95)
*   :bug: Fixes small bug in test_policy utility [`de6a15a`](https://github.com/rickstaa/stable-learning-control/commit/de6a15a45d6c9b129115538da84fe1b33b782168)
*   :twisted_rightwards_arrows: Merge branch 'add_torch_tensorboard_support' into main [`8d324d1`](https://github.com/rickstaa/stable-learning-control/commit/8d324d1b4c0d7671b304514984f0347c4f59a2fe)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`9d30767`](https://github.com/rickstaa/stable-learning-control/commit/9d30767eb4ccc1d6c374b4aabd57ec03ae621e7f)
*   :bookmark: Updates code version to v0.3.0 [`75c69c4`](https://github.com/rickstaa/stable-learning-control/commit/75c69c4f6500ffe823ca638bdaa5af493fc70f4b)
*   :bookmark: Updates documentation version to v0.2.9 [`7fd9458`](https://github.com/rickstaa/stable-learning-control/commit/7fd945859ee8557f41de47d74e85b19665af71af)
*   :green_heart: Adds PAT to autotag action such that it triggers other workflows [`b505bd3`](https://github.com/rickstaa/stable-learning-control/commit/b505bd3bcfe6bb5794597ec5d45a8749942327ad)
*   :wrench: Updates Torch hyperparameter defaults [`c8dd11e`](https://github.com/rickstaa/stable-learning-control/commit/c8dd11e038e5433a3463cceb24bfcb9cdcef7b08)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`3115913`](https://github.com/rickstaa/stable-learning-control/commit/3115913977a64f8648489050eb1d76ccf60f81c2)
*   :bookmark: Updates code version to v0.5.0 [`4b68c87`](https://github.com/rickstaa/stable-learning-control/commit/4b68c873cae1ffefdbafda3bef0e75dc89b9d27f)
*   :bookmark: Updates documentation version to v0.2.11 [`021bbb6`](https://github.com/rickstaa/stable-learning-control/commit/021bbb686530067577479a7a5fc2e2cfad2d4de5)
*   :green_heart: Fixes release action tagging behavoir [`33b459e`](https://github.com/rickstaa/stable-learning-control/commit/33b459e68b1c21e75c90ab32d00d0d08a8b23e71)
*   :bug: Fixes a pip installation bug [`cd068c3`](https://github.com/rickstaa/stable-learning-control/commit/cd068c3b66add4831a87a683590a70fbb8bf24f0)
*   :wrench: Updates release changelog configuration file [`9ff2884`](https://github.com/rickstaa/stable-learning-control/commit/9ff2884e8d68e2176e4e6f79754dabd2648aaa21)
*   :art: Updates code formatting [`7248b7b`](https://github.com/rickstaa/stable-learning-control/commit/7248b7bd6ef18e40e552064d2900eaa85cee499e)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`f3928e3`](https://github.com/rickstaa/stable-learning-control/commit/f3928e35d64415a24b134ddd17da3fd61ea6816d)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`73c5494`](https://github.com/rickstaa/stable-learning-control/commit/73c54945e7683920ece91682697e4756f0850638)
*   :green_heart: Updates release github-action [`7bd6471`](https://github.com/rickstaa/stable-learning-control/commit/7bd6471703c27dc86f122508030b98d2024e90d7)
*   :memo: Updates CHANGELOG.md [`32e735a`](https://github.com/rickstaa/stable-learning-control/commit/32e735a49ba2589201c08a9077e2da9cddffe953)
*   :memo: Updates CHANGELOG.md [`51a3492`](https://github.com/rickstaa/stable-learning-control/commit/51a349204521c50f483ec4971ee824b446c931cf)
*   :green_heart: Fixes release github action [`8a45522`](https://github.com/rickstaa/stable-learning-control/commit/8a45522844953cac1c6d461441966aca8a3d3a09)
*   :memo: Updates CHANGELOG.md [`ed54568`](https://github.com/rickstaa/stable-learning-control/commit/ed545689e69ac03315d368a27183cdced7259cc0)
*   :green_heart: Fixes release workflow [`8d7b851`](https://github.com/rickstaa/stable-learning-control/commit/8d7b85195531436abca126d618421edf234fc576)
*   :green_heart: Adds pre-release action [`745f4eb`](https://github.com/rickstaa/stable-learning-control/commit/745f4eb558c7217d3d6ee2d9c7bb4950473a5d97)
*   :green_heart: Fixes bug in release gh-action: [`da2db8a`](https://github.com/rickstaa/stable-learning-control/commit/da2db8a9ddc573e72d20442d0a984a3a49f9d1f6)
*   :green_heart: Fixes github release action [`20bc3e3`](https://github.com/rickstaa/stable-learning-control/commit/20bc3e3af866dda1b3e3e4e77e04370ab1a26ea7)
*   :green_heart: Fixes release gh-action [`0d3be6a`](https://github.com/rickstaa/stable-learning-control/commit/0d3be6a8b7438c51af307e6f3d106af39317238b)
*   :memo: Updates CHANGELOG.md [`c0c06d8`](https://github.com/rickstaa/stable-learning-control/commit/c0c06d8b08d469670e1c5637a20020c4f9541054)
*   :memo: Updates CHANGELOG.md [`e5d2ea2`](https://github.com/rickstaa/stable-learning-control/commit/e5d2ea236be71f16e78b1e178ae3958dc6e2d969)
*   :memo: Updates CHANGELOG.md [`a32d5d6`](https://github.com/rickstaa/stable-learning-control/commit/a32d5d6e0cf39501bbfbe705365e16ee74a74bdc)
*   :memo: Updates CHANGELOG.md [`3279843`](https://github.com/rickstaa/stable-learning-control/commit/32798433d45c5b12262d38b9d8f6612381300e50)
*   :green_heart: Fixes bug inside the release gh-action [`4c81a1e`](https://github.com/rickstaa/stable-learning-control/commit/4c81a1e522a17bc301128d0aa96104244617ca8b)
*   :green_heart: Updates gh-actions [`f1a7a71`](https://github.com/rickstaa/stable-learning-control/commit/f1a7a71b7c5186684aba5ef9a61e32b09d8da6d0)
*   :green_heart: Fixes release github action [`22830dd`](https://github.com/rickstaa/stable-learning-control/commit/22830dde021962239bbce9c31c36f0dc77354000)
*   :green_heart: Disables action-update semver [`cb05709`](https://github.com/rickstaa/stable-learning-control/commit/cb05709acacfcf250b2e9322739732c2f3080f39)
*   :green_heart: Fixes release gh-action [`abc7a33`](https://github.com/rickstaa/stable-learning-control/commit/abc7a33f61512fa257a827e2b60cb95fbf69a32f)
*   :memo: Updates CHANGELOG.md [`7285820`](https://github.com/rickstaa/stable-learning-control/commit/7285820b92f80843a9ac5b95aad654be4faea5f2)
*   :bookmark: Updates documentation version to v0.2.7 [`66c606d`](https://github.com/rickstaa/stable-learning-control/commit/66c606d39bef393e08571fa7b2e2275498dd8a41)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ca22707`](https://github.com/rickstaa/stable-learning-control/commit/ca227077eee983a10cd6f1ebd7f7df94986e8683)
*   :green_heart: Fixes gh release action detached head bug [`69471ac`](https://github.com/rickstaa/stable-learning-control/commit/69471acf89bf9e3d50ae06559f1b215ed1020094)
*   :bookmark: Updates documentation version to v0.2.6 [`43a55a5`](https://github.com/rickstaa/stable-learning-control/commit/43a55a59f0697e9c12b05b4d52c602344d572326)
*   :bookmark: Updates documentation version to v0.2.4 [`7966ff5`](https://github.com/rickstaa/stable-learning-control/commit/7966ff5327943ac9cdcce8549b0e541e154f836e)
*   :bookmark: Updates documentation version to v0.2.3 [`5889c97`](https://github.com/rickstaa/stable-learning-control/commit/5889c971e395e5cb17fd42dcefdb1e57a8483a09)
*   :bookmark: Updates documentation version to v0.2.2 [`c387560`](https://github.com/rickstaa/stable-learning-control/commit/c3875609a871ff8d3fa140be2623de421c96d9c1)
*   :bookmark: Updates documentation version to v0.2.0 [`d9fd0e2`](https://github.com/rickstaa/stable-learning-control/commit/d9fd0e29acec528dd1fa775b39aad51a6f7e153b)
*   :bookmark: Updates documentation version to v0.2.1 [`fbc8adc`](https://github.com/rickstaa/stable-learning-control/commit/fbc8adcc0af1a90aa764a509db699f843932f359)
*   :bookmark: Updates documentation version to v0.1.6 [`26032b2`](https://github.com/rickstaa/stable-learning-control/commit/26032b2ea88ec35cf3e532da8f7ae267d31cab00)
*   :bookmark: Updates documentation version to v0.1.5 [`80aee7a`](https://github.com/rickstaa/stable-learning-control/commit/80aee7a1e7f78579c8da8a604d383e08da56bdc1)
*   :rewind: Revert ":bookmark: Updates documentation version: v0.1.5 → v0.1.5" [`284f438`](https://github.com/rickstaa/stable-learning-control/commit/284f4381d4bb36c82be653f59490dacea02ff947)
*   :bookmark: Updates documentation version: v0.1.5 → v0.1.5 [`06dce19`](https://github.com/rickstaa/stable-learning-control/commit/06dce19b50cf3bdd8dcc1cee4595ea5986ceef8b)
*   :green_heart: Fixes syntax error in release gh-action [`612e5ea`](https://github.com/rickstaa/stable-learning-control/commit/612e5ea117c9344771620f6da9c65eed481e7774)
*   :green_heart: Fixes changelog gh-action [`5f254d2`](https://github.com/rickstaa/stable-learning-control/commit/5f254d2a374f1c4df3c4b8794488af7044361508)
*   :green_heart: Fixes release gh action [`0dabacf`](https://github.com/rickstaa/stable-learning-control/commit/0dabacf78d409de0c5138f277ad04f20df1c08e3)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`9decccf`](https://github.com/rickstaa/stable-learning-control/commit/9decccf86248c292945fda3ac31bfbf505396adf)
*   :green_heart: Fixes version release job [`4c0011f`](https://github.com/rickstaa/stable-learning-control/commit/4c0011f84ecbeba795cbf8f460edfd2bf7979565)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`18afc55`](https://github.com/rickstaa/stable-learning-control/commit/18afc554d876406bc67a357343b6c3e5499cb409)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`f231c13`](https://github.com/rickstaa/stable-learning-control/commit/f231c131cfa3b46d5176b2679721235786453b6f)
*   :bug: Fixes mac github action name [`ca563f7`](https://github.com/rickstaa/stable-learning-control/commit/ca563f7b2d52b86fab4b7edc8be22e53e1b780ef)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6104eef`](https://github.com/rickstaa/stable-learning-control/commit/6104eefb1d1f5a91989f4457d86b2bce6732cb55)
*   :memo: Updates the documentation [`7544d75`](https://github.com/rickstaa/stable-learning-control/commit/7544d75c4583ae94f64daa32b420338d40ac47d9)
*   :green_heart: Fixes bug in release gh-action [`6067635`](https://github.com/rickstaa/stable-learning-control/commit/60676351b3821c4ed6be725e2570209a21562788)
*   :green_heart: Adds gh-action test data [`840bd04`](https://github.com/rickstaa/stable-learning-control/commit/840bd0415f3a558d3734c601b2508f563924cf4d)
*   :green_heart: Updates docs and release gh actions [`3cff977`](https://github.com/rickstaa/stable-learning-control/commit/3cff9779bf653d542515170804418ae20619d339)
*   :art: Format Python code with psf/black push [`c512828`](https://github.com/rickstaa/stable-learning-control/commit/c512828394574a103c106a89d7e4455fe4bcfa63)
*   :fire: Removes gh-action test data [`135646b`](https://github.com/rickstaa/stable-learning-control/commit/135646b898fe957945af1a625ea5ac82c6089ef7)
*   :green_heart: Adds auto-changelog and bumpversion gh-actions [`f79e7db`](https://github.com/rickstaa/stable-learning-control/commit/f79e7db5ba0768c4405d7daf358b8bcfffd8a097)
*   :green_heart: Fixes quality check gh-action [`0fb043a`](https://github.com/rickstaa/stable-learning-control/commit/0fb043a9238c154656f88d1b31afb9ce68a7b119)
*   :goal_net: Catch errors to allow autoformatting [`4c78b41`](https://github.com/rickstaa/stable-learning-control/commit/4c78b414a6126cbe15eb5c1d8c7074aaffbb477f)
*   :green_heart: Updates gh-action [`07e36b7`](https://github.com/rickstaa/stable-learning-control/commit/07e36b7d62ef87f8ffbd85b5cc691d58d4ee5c35)
*   :green_heart: Enables code quality gh-actions [`2e1ed92`](https://github.com/rickstaa/stable-learning-control/commit/2e1ed92459739321670d7c48e2548df9877031f2)
*   :green_heart: Updates github actions [`76a9f86`](https://github.com/rickstaa/stable-learning-control/commit/76a9f86ead050d7c19700e76959a51522836b7c6)
*   :green_heart: Updates code quality gh-actions [`16591b4`](https://github.com/rickstaa/stable-learning-control/commit/16591b43623ea2fc3eca0dc37c171540203fc064)
*   :green_heart: Fixes release gh-action [`5e1d376`](https://github.com/rickstaa/stable-learning-control/commit/5e1d376a05950c1d519585c551b86df58da8849b)
*   :green_heart: Update docs check github action. [`d293b42`](https://github.com/rickstaa/stable-learning-control/commit/d293b425702d758c00dc7bf8559442f5bc92f041)
*   :green_heart: Updates release gh-action [`f966ee1`](https://github.com/rickstaa/stable-learning-control/commit/f966ee1044b137754c72ba2b15f2327d3e5cbdab)
*   :green_heart: Updates docs gh action [`3538689`](https://github.com/rickstaa/stable-learning-control/commit/35386898d75341969a56d6f31841347054e12335)
*   :green_heart: Fixes output bug in release gh-action [`59cc86f`](https://github.com/rickstaa/stable-learning-control/commit/59cc86fcdbb5e0e1d186924725903db3a01eed86)
*   :white_check_mark: Test out pull request creation action [`9bb87d2`](https://github.com/rickstaa/stable-learning-control/commit/9bb87d2f812ae4406be6478c10932058d0305307)
*   :green_heart: Updates gh-actions [`78c476a`](https://github.com/rickstaa/stable-learning-control/commit/78c476a26c48528bb59f09f9e10923d537266992)
*   :bug: Updates python version inside the gh-actions [`8ffa382`](https://github.com/rickstaa/stable-learning-control/commit/8ffa3826bc1609c2924af8b2c39ded3add391375)
*   :bookmark: Updates documentation versioning [`583393c`](https://github.com/rickstaa/stable-learning-control/commit/583393c413e63df64e370d8b45f2eff9b3e56751)
*   :green_heart: Updates code quality gh-action syntax [`14497ad`](https://github.com/rickstaa/stable-learning-control/commit/14497ad148c1a5a0fb9eb8d285b68fa973c7b43a)
*   :green_heart: Fixes release gh action [`c03c72b`](https://github.com/rickstaa/stable-learning-control/commit/c03c72bf5b01dc893ee4eb12eb93b9defbb6b97d)
*   :green_heart: Fixes bug in release gh-action [`483ad95`](https://github.com/rickstaa/stable-learning-control/commit/483ad95a3af8dce729f86cca9c2ca76323a6bd4c)
*   :green_heart: Fixes release gh-action commit step [`a9dfb68`](https://github.com/rickstaa/stable-learning-control/commit/a9dfb68276c0b0c5c486bf888c5ea1e064e33c1b)
*   :green_heart: Fixes release gh action [`6913c08`](https://github.com/rickstaa/stable-learning-control/commit/6913c089a45936f3181775e6cb2fbbf6a4cce59d)
*   :bug: Fixes bug in gh docs publish workflow [`234e2b8`](https://github.com/rickstaa/stable-learning-control/commit/234e2b85feb6839fdfa26b15fdadbc1ba77b15d9)
*   :green_heart: Fixes release gh-action [`3578002`](https://github.com/rickstaa/stable-learning-control/commit/3578002e09103e93fb3f16b37963b4cd3d75cb57)
*   :green_heart: Updates mlc release and docs release gh-actions [`1b10a48`](https://github.com/rickstaa/stable-learning-control/commit/1b10a48ddde5d3fba96dd91033b9dae120ecc0bb)
*   Fixes syntax bug in gh-action [`8501fb6`](https://github.com/rickstaa/stable-learning-control/commit/8501fb6d94453305770856d216f74238c0d67696)
*   :memo: Change README.md buttons [`8041819`](https://github.com/rickstaa/stable-learning-control/commit/8041819a5595b2e3b522c73a9ec83f9ca3c954da)
*   :green_heart: Fixes gh release action branch name [`84dad7e`](https://github.com/rickstaa/stable-learning-control/commit/84dad7ebc95a6233face3d9d60f51f5c128c953f)
*   :green_heart: Fixes release gh-action [`1f68e88`](https://github.com/rickstaa/stable-learning-control/commit/1f68e8851554e84582cccc634830f2a269091661)
*   :green_heart: Fixes other bug in release github-action [`5dfa2de`](https://github.com/rickstaa/stable-learning-control/commit/5dfa2de0212608b5518c066d410772b05248f8c5)
*   :green_heart: Fixes release gh-action [`41191c6`](https://github.com/rickstaa/stable-learning-control/commit/41191c634033c7675eb72c8949cd9831dbf64131)
*   :green_heart: Updates release gh-action [`bf43239`](https://github.com/rickstaa/stable-learning-control/commit/bf43239413fa439b31b77c2e29bc1561977416d7)
*   :memo: Updates documentation [`960da39`](https://github.com/rickstaa/stable-learning-control/commit/960da398e0a794e19e2646d79ee8b13e58491412)
*   :memo: Updates documentation [`1a89225`](https://github.com/rickstaa/stable-learning-control/commit/1a892255037ae355748bb94fcbcc91e03c00c28d)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`ac3bcd8`](https://github.com/rickstaa/stable-learning-control/commit/ac3bcd8bd4dabd8f3e3700b48e38ee314a588a87)
*   :green_heart: Fixes syntax error inside github action [`18728b8`](https://github.com/rickstaa/stable-learning-control/commit/18728b84dbe6f096f83d07818ee9556c6de12bf9)
*   :white_check_mark: Enables pre-release workflow [`6054e7d`](https://github.com/rickstaa/stable-learning-control/commit/6054e7de78d0997e00c5574f779fd53ad9e8b571)
*   :green_heart: Force the python cache to be rebuild [`d6172d5`](https://github.com/rickstaa/stable-learning-control/commit/d6172d5f7c2892a7fdbf07e18f0d6c116f6f82cb)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`b20f329`](https://github.com/rickstaa/stable-learning-control/commit/b20f329875df594928636277370328f5a5192dd6)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`6e6f3c8`](https://github.com/rickstaa/stable-learning-control/commit/6e6f3c8a3acdd01222541dc29164e771224ca975)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`793cf6a`](https://github.com/rickstaa/stable-learning-control/commit/793cf6a70514c5926ad647375790f65ff7302316)
*   :memo: Updates documentation [`5949d97`](https://github.com/rickstaa/stable-learning-control/commit/5949d9751ca18ca195a0cbbe13fb23e5cd550755)
*   :memo: Updates docs structure [`6bffce6`](https://github.com/rickstaa/stable-learning-control/commit/6bffce6b2464b59ef6569cbb87b97f5eb34af99f)
*   :memo: Adds the documentation [`96c1417`](https://github.com/rickstaa/stable-learning-control/commit/96c1417b24a0eff2a7fd4bca4d47ab6f43d5dac1)
*   :memo: Updates documentation and add documentation CI action [`ed40e0e`](https://github.com/rickstaa/stable-learning-control/commit/ed40e0eea76d2eee9541663dfb9ae83edd4dda8c)
*   :bug: Fixes syntax errors [`fa0d646`](https://github.com/rickstaa/stable-learning-control/commit/fa0d6467bbb089ce7dfc44c9ce4362d0fa965f22)
*   :fire: Removes some unused files and adds some Useful scripts" [`52958b4`](https://github.com/rickstaa/stable-learning-control/commit/52958b4680f7945da1ab0a4c9e2ec1d367fdb871)
*   :bug: Fixes some bugs in the LAC algorithm and updates the documentation [`b73a99b`](https://github.com/rickstaa/stable-learning-control/commit/b73a99bcc7c83c1c6b14dae0941b17f101ff88db)
*   :memo: Updates errors in the documentation. [`d2ac20c`](https://github.com/rickstaa/stable-learning-control/commit/d2ac20c3b205874221ced23e075ab4e31341431d)
*   :art: Improves code syntax [`bcc676c`](https://github.com/rickstaa/stable-learning-control/commit/bcc676cfa3d97efd70b861252a8c222afd340d77)
*   :construction_worker: Adds linux/mac/windows ci checks [`159d8d9`](https://github.com/rickstaa/stable-learning-control/commit/159d8d97dd05783b230197db5e684f80f133bdb8)
*   :green_heart: Perform some CI tests [`bd10542`](https://github.com/rickstaa/stable-learning-control/commit/bd1054296ff23471602b2ac73fd99cffc34c672b)
*   :fire: Removes redundant folders [`97cf08c`](https://github.com/rickstaa/stable-learning-control/commit/97cf08c3e625f4241478b47568f3cbbbc2708cfe)
*   :memo: Cleans up documentation [`9376e27`](https://github.com/rickstaa/stable-learning-control/commit/9376e2780d0489dcd385f99acfa227d335503004)
*   :green_heart: Updates gh-actions [`1ed50ba`](https://github.com/rickstaa/stable-learning-control/commit/1ed50ba98b285b24cb6bf8b77a5ed4d7303a2a1a)
*   :bug: Disables broken CI script [`d2f3dc4`](https://github.com/rickstaa/stable-learning-control/commit/d2f3dc449f012ba6881432f291897c528a22408f)
*   :construction_worker: Re-adds docs CI action [`ae324ed`](https://github.com/rickstaa/stable-learning-control/commit/ae324edf777aaf7c034c85192c2624c552782502)
*   :green_heart: Fixes bug in the docs ci action [`8426c92`](https://github.com/rickstaa/stable-learning-control/commit/8426c922de47e036a12d9e0d35e6f4a118c25ea2)
*   :memo: Updates Changelog [`de02746`](https://github.com/rickstaa/stable-learning-control/commit/de027469761da81d7702622bbf82eea12d206527)
*   :bug: Fixes simzoo module import errors [`0c172b0`](https://github.com/rickstaa/stable-learning-control/commit/0c172b096e66e3350cb09285c7a306c2167b1926)
*   :memo: Updates README.md [`e81a648`](https://github.com/rickstaa/stable-learning-control/commit/e81a6488e183696fafd349d613f45c7b357f6557)
*   :construction_worker: Adds docs build check step [`6afd063`](https://github.com/rickstaa/stable-learning-control/commit/6afd063d7a1b1dd56daf6b83bf1657cdc802c996)
*   :construction_worker: Make ci spellcheck less strict [`8deebb2`](https://github.com/rickstaa/stable-learning-control/commit/8deebb2876f71f7187faf0a1b523c28d4726ff57)
*   :construction_worker: Fixes github pytest action [`40492b5`](https://github.com/rickstaa/stable-learning-control/commit/40492b583318c849cb298e565534d5c425602f2c)
*   :bug: Fixes github actions cache step [`e57ea50`](https://github.com/rickstaa/stable-learning-control/commit/e57ea50d18e167b158142792db5e3db9b354b22b)
*   :construction_worker: Updates CI action python version [`a29e99e`](https://github.com/rickstaa/stable-learning-control/commit/a29e99ec05090b9674343f0ec008ed9639f85402)
*   :memo: Updates bug_report.md [`aeb5e9b`](https://github.com/rickstaa/stable-learning-control/commit/aeb5e9becee401399ae1272ab762d5b59c44360b)
*   :green_heart: Fixes ci build [`90d6f74`](https://github.com/rickstaa/stable-learning-control/commit/90d6f74ff3fa6845142dfa1e9083bf74825932be)
*   :memo: Updates control readme.md [`2a966b5`](https://github.com/rickstaa/stable-learning-control/commit/2a966b55e18cd5e08ac59fc87ff4ffd276a7f22c)
*   :memo: Adds os CI badges to the readme [`c0abc05`](https://github.com/rickstaa/stable-learning-control/commit/c0abc054bad2056f22d3867331dde8f85c62ab55)
*   :green_heart: Updates test gh-action [`6497f23`](https://github.com/rickstaa/stable-learning-control/commit/6497f23584a6d66e1790791b2df0a0690a344430)
*   :memo: Updates main README.md [`1b8634e`](https://github.com/rickstaa/stable-learning-control/commit/1b8634ecf1bf58015da8d95448c40e51367b27a5)
*   :bookmark: Bump version: 0.0.1 → 0.1.0 [`5bf3396`](https://github.com/rickstaa/stable-learning-control/commit/5bf3396a231f86de64736f47fcb93885f08e8823)
*   :construction: Adds dummy function to the tf2 lac template [`574f874`](https://github.com/rickstaa/stable-learning-control/commit/574f874bb6328d5d882a4f8179f2faac9e28c342)
*   :bookmark: Bump version: 0.1.0 → 0.1.1 [`f25db48`](https://github.com/rickstaa/stable-learning-control/commit/f25db48a9114dbc5db50f26f43f2c8ab242db9d0)
*   :memo: Updates python version support badge [`4a94070`](https://github.com/rickstaa/stable-learning-control/commit/4a94070d5a0a41caef0a297820f4ce4aabbd54dd)
*   :green_heart: Updates gh-action tests [`8a8f41b`](https://github.com/rickstaa/stable-learning-control/commit/8a8f41bbf8826bf79d7752c3c5735f2d334898d0)
*   :green_heart: Another CI docs release fix [`a7ba90f`](https://github.com/rickstaa/stable-learning-control/commit/a7ba90f23b2687f7a70402e50bb99836059bcd6d)
*   :construction_worker: Fixes github actions syntax error [`6bf75dc`](https://github.com/rickstaa/stable-learning-control/commit/6bf75dc3d4ea044205144fdca376d6d78849ca5d)
*   :construction_worker: Fixes flake8 action repository url [`5dd9c0c`](https://github.com/rickstaa/stable-learning-control/commit/5dd9c0c36caac596c7e11977f5695c1e3fbf777f)
*   :bug: Fixes mlc github action bug [`718a276`](https://github.com/rickstaa/stable-learning-control/commit/718a276e0ac6d5ca991f3cfbd46c1711fe3bcf47)
*   :memo: Updates the os badge urls [`33546b1`](https://github.com/rickstaa/stable-learning-control/commit/33546b17bec3e3757d414726e1017ca05c31122a)
*   :wrench: Updates bumpversion config [`21d3010`](https://github.com/rickstaa/stable-learning-control/commit/21d3010ea5f40b622934c3da635fb168263d275a)
*   :green_heart: Updates gh-action tests [`5014392`](https://github.com/rickstaa/stable-learning-control/commit/50143925b9993795f1aeb4e8b2e97e6069c67bb3)
*   :green_heart: Adds documentation artifacts [`0444a29`](https://github.com/rickstaa/stable-learning-control/commit/0444a291eef627f46d8a22622719fa247035bd9d)
*   :construction_worker: Change python cache file [`721dad5`](https://github.com/rickstaa/stable-learning-control/commit/721dad5eb552da956b7386508bee450672546dc8)
*   :memo: Updates documentation todos [`7ad1116`](https://github.com/rickstaa/stable-learning-control/commit/7ad1116e946a53800b112831a2f05defd21b37de)
*   :bug: Fixes github actions pip upgrade step [`753b53c`](https://github.com/rickstaa/stable-learning-control/commit/753b53cc473d73a551c0f21b5e3119361376b73b)
*   :green_heart: Changes the gh action tests [`ee8f49e`](https://github.com/rickstaa/stable-learning-control/commit/ee8f49e0d3afe4a088f2ec5bf604cf4dfeb3383a)
*   :construction_worker: Force github actions to rebuild pip cache [`0449167`](https://github.com/rickstaa/stable-learning-control/commit/04491671f87f45d3a6e865c0a7590c7ae1df7df2)
*   :wrench: Updates bumpversion config [`8d6c4d0`](https://github.com/rickstaa/stable-learning-control/commit/8d6c4d094e41f48a266d5cb3de11ac3793948c1f)
*   :green_heart: Fixes bug in the docs release github action [`8703507`](https://github.com/rickstaa/stable-learning-control/commit/870350776c552702fe86f112fca0bc309aaebd62)
*   :bug: Fixes pytest github action command [`768be0a`](https://github.com/rickstaa/stable-learning-control/commit/768be0aa5981f199fefb0af7eb87993380fc63bd)
*   :sparkles: Re-adds simzoo submodule [`8784a2a`](https://github.com/rickstaa/stable-learning-control/commit/8784a2ab2a64b652d744fe7dc4ab329a73685e08)
*   :memo: Updates documentation link in the README.md [`db90671`](https://github.com/rickstaa/stable-learning-control/commit/db906712c95ceebfcd265567c41f1fa326291599)
*   :green_heart: Fixes syntax error in github action [`21776e7`](https://github.com/rickstaa/stable-learning-control/commit/21776e7cacf3ca49033526d11bf0d6b751e65f1f)
*   :construction_worker: Updates CI config files [`45ebf3a`](https://github.com/rickstaa/stable-learning-control/commit/45ebf3a5e8f046a6be073b458cfb097389968f2b)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`a5aeedd`](https://github.com/rickstaa/stable-learning-control/commit/a5aeedd8be2a45a2ef77f8089f0c9a622f2a3938)
*   :construction_worker: Updates github actions checkout token [`30f4ab6`](https://github.com/rickstaa/stable-learning-control/commit/30f4ab65ce11a48c5e5467f02e927a4333ac5427)
*   :green_heart: Updates docs ci action to create orphan branch [`c8550ee`](https://github.com/rickstaa/stable-learning-control/commit/c8550ee4101b7b2f26197772269da272ff90cac6)
*   :construction_worker: Changes the documentation ci name [`80c9049`](https://github.com/rickstaa/stable-learning-control/commit/80c9049c3058c1245997d4ec9b0b74a25b6e8524)
*   :bug: Fixes a small ci bug [`5cb1437`](https://github.com/rickstaa/stable-learning-control/commit/5cb14375f76cfec82ac390a5b6616b744c607319)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`4688716`](https://github.com/rickstaa/stable-learning-control/commit/4688716d725a1aa10005a6609c1712f3cc327f86)
*   :green_hearth: Adds different tag recognition github action [`e99e4f2`](https://github.com/rickstaa/stable-learning-control/commit/e99e4f219d0571e7733bb675cf0e7c593abbc776)
*   :green_heart: Adds some gh action tests [`92353e5`](https://github.com/rickstaa/stable-learning-control/commit/92353e5ee7c2669c38d4d7fef4741c6056c4a8bf)
*   :green_heart: Updates gh-actions tests [`cb2e4de`](https://github.com/rickstaa/stable-learning-control/commit/cb2e4de6f418ab622a196592f5f73dc6ff91b3a5)
*   :green_heart: Updates gh action tests [`d8b454a`](https://github.com/rickstaa/stable-learning-control/commit/d8b454a66c99e17dc4a05921f61a0fb7304992e0)
*   :green_heart: Removes redundant gh action step [`5ffb682`](https://github.com/rickstaa/stable-learning-control/commit/5ffb682d0d78314a86a09f6b0f17ffa95f74ca44)
*   :green_heart: Updates gh-actions tests [`01e1643`](https://github.com/rickstaa/stable-learning-control/commit/01e1643135d5e64669621dd2d169ed98231cc9d9)
*   :green_heart: Updates gh-actions tests [`88da879`](https://github.com/rickstaa/stable-learning-control/commit/88da8791beda88bb255565572a8355abcdfaf4a4)
*   :green_heart: Fixes docs ci bug [`c1e7b69`](https://github.com/rickstaa/stable-learning-control/commit/c1e7b69af5b95682860542743bb2ffca2a995d16)
*   :bug: Fixes a small bug in the docs release ci action [`6548c52`](https://github.com/rickstaa/stable-learning-control/commit/6548c5249521e0c73aff2241761e666c08913254)
*   :art: :wrench: Updates bumpversion config [`aa404ea`](https://github.com/rickstaa/stable-learning-control/commit/aa404eac2adfef613aac14042c5445dd3fc1b7c2)
*   :bookmark: Updates version [`3ce4702`](https://github.com/rickstaa/stable-learning-control/commit/3ce47025e1d2ddbe3cce390608a2caa79059fc02)
*   :green_heart: Updates gh-actions tests [`daff84b`](https://github.com/rickstaa/stable-learning-control/commit/daff84b40dea4c5c13ad8817112c66d6c35273ea)
*   :green_heart: Fixes docs ci build [`3657569`](https://github.com/rickstaa/stable-learning-control/commit/36575697c177653a844428c5613cc0b0a73bfc62)
*   :green_heart: Updates new docs github action [`48856ec`](https://github.com/rickstaa/stable-learning-control/commit/48856ec91980e4b7908baeafb26b8e72a6d15099)
*   :wrench: Updates sphinx config file [`ed4a5c8`](https://github.com/rickstaa/stable-learning-control/commit/ed4a5c87b0d4d0adc57cde01e8cee352ecee4566)
*   :construction_worker: Fixes flake8 fail on error problem [`baa3332`](https://github.com/rickstaa/stable-learning-control/commit/baa333220d997774db465844956cbe01fd4e62d4)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`3347c98`](https://github.com/rickstaa/stable-learning-control/commit/3347c9861048c68f356d0a9019b88cb4c9895256)
*   :construction_worker: Adds github token to ci action [`50f471d`](https://github.com/rickstaa/stable-learning-control/commit/50f471de08cd696387c63da8df47296a8aa9a230)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine-learning-control into main [`add11ee`](https://github.com/rickstaa/stable-learning-control/commit/add11ee210be0ae1953d2b2967abea3b864e6b41)
*   :bug: Fixes bug in .gitmodules file [`5493f2f`](https://github.com/rickstaa/stable-learning-control/commit/5493f2f0d69729a96e0ca26f0aa6fb4fa8998622)
*   :wrench: Updates gitmodules file to accept relative urls [`4ad5624`](https://github.com/rickstaa/stable-learning-control/commit/4ad5624fd89c5b50a05b32d96860640ff51f34d2)
*   :memo: Updates the README.md formatting [`43a96a7`](https://github.com/rickstaa/stable-learning-control/commit/43a96a76102e2486d686bb5b6b152e994f693830)
*   :green_heart: Forces pip cache to rebuild [`d798cb9`](https://github.com/rickstaa/stable-learning-control/commit/d798cb9fd7003f36e400ee412bbcec1423de3b52)
*   :green_heart: Changes docs ci commit message [`81cd922`](https://github.com/rickstaa/stable-learning-control/commit/81cd9229e8167826229c6557da94935e62fd8ccb)
*   :wrench: Adds .nojekyll file to documentation build step [`b5c50c6`](https://github.com/rickstaa/stable-learning-control/commit/b5c50c6d5b76a22c7f0c1131988e02ecbfa595ab)
*   :twisted_rightwards_arrows: Merge branch 'renovate/cloudpickle-1.x' into main [`59af010`](https://github.com/rickstaa/stable-learning-control/commit/59af0106da22af2f975f1b4b07243ce632a5c19a)
*   :twisted_rightwards_arrows: Merge branch 'dependabot/pip/docs/tensorflow-2.4.0' into main [`56548ca`](https://github.com/rickstaa/stable-learning-control/commit/56548ca544966b22958dabdb07f537d43c30a8a8)
*   :fire: Removes redundant submodule [`ab868cd`](https://github.com/rickstaa/stable-learning-control/commit/ab868cd4571c835b97c14716e9e7931b4c9eeadd)
*   :twisted_rightwards_arrows: Merge branch 'main' into dependabot/pip/docs/tensorflow-2.4.0 [`df9aa73`](https://github.com/rickstaa/stable-learning-control/commit/df9aa7397a362d456d0db93cb5f6967457b104fe)
*   :twisted_rightwards_arrows: Merge branch 'main' into renovate/cloudpickle-1.x [`6de29ab`](https://github.com/rickstaa/stable-learning-control/commit/6de29ab3a2de1102b52825d7e54a6223ce743e8c)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`a27a0f3`](https://github.com/rickstaa/stable-learning-control/commit/a27a0f3eb0f154663cb612723d6fee414741b2d2)
*   :wrench: Updates setup.py and documentation [`ecba2c7`](https://github.com/rickstaa/stable-learning-control/commit/ecba2c786b0cb43d181c01896a19aa23082e79a3)
*   :green_heart: Updates github actions [`3a85ad4`](https://github.com/rickstaa/stable-learning-control/commit/3a85ad4c2c728d35d0d34938c66320e34b70613d)
*   :green_heart: Updates github actions [`a6a7c28`](https://github.com/rickstaa/stable-learning-control/commit/a6a7c28e70749e03134693e3d21b560589e8e798)
*   :wrench: Migrate to pyproject.toml (PEP518/517) [`e564b2e`](https://github.com/rickstaa/stable-learning-control/commit/e564b2ea06ada16e5988d0ccbe5ed56bee0ad7eb)
*   :bug: Fixes CI config bugs [`d9da3fd`](https://github.com/rickstaa/stable-learning-control/commit/d9da3fd0ba7b9b61659b7ebfe97b201313f4e97e)
*   :bug: Test pip cache [`85f1380`](https://github.com/rickstaa/stable-learning-control/commit/85f1380a99d0498ad14a64e3d47400695e7aa2e0)
*   :recycle: Cleanup CI config file [`538ae20`](https://github.com/rickstaa/stable-learning-control/commit/538ae2086ae1b616b5be0523a2f7abc34c96700d)
*   :bug: Fixes a bug that was introduced due to e564b2ea06 [`a03c2d0`](https://github.com/rickstaa/stable-learning-control/commit/a03c2d0bab9a637f954ba271a925a3e92c8fc56c)
*   :fire: Removes redundant action files [`4b499ee`](https://github.com/rickstaa/stable-learning-control/commit/4b499eef35f0175954ddfef94ce5c866d4f35f56)
*   :fire: Removes redundant files and updates dev reqs [`97f1f6d`](https://github.com/rickstaa/stable-learning-control/commit/97f1f6d5768b2c0e88e87b51e81fb6b49bbb8362)
*   :bug: Updates pytests [`943badf`](https://github.com/rickstaa/stable-learning-control/commit/943badfeeeb134fa9db9d29696203f9e309e0625)
*   :wastebasket: Removes depricated code and updates CI config [`b2f4187`](https://github.com/rickstaa/stable-learning-control/commit/b2f4187fc16efe0cfcddf0994dcef2c43d996f41)
*   :wrench: Add security config [`f66d4c5`](https://github.com/rickstaa/stable-learning-control/commit/f66d4c58fe305cfa117d2bb095dd7a97560c2bb6)
*   :fire: Removes redundant requirements file [`d5a5436`](https://github.com/rickstaa/stable-learning-control/commit/d5a543626d9c21742e965e5e5154433d0746a08b)
*   :bug: Removes redundant CI config if statements [`d3acd62`](https://github.com/rickstaa/stable-learning-control/commit/d3acd62f3c2683736b25d9061a37dd3ad06a1847)
*   :wrench: Updates CI config [`aeca3cf`](https://github.com/rickstaa/stable-learning-control/commit/aeca3cf8f5a2a57712c4094d14fcd33670fe56fa)
*   :bug: Fixes CI config [`94335d8`](https://github.com/rickstaa/stable-learning-control/commit/94335d81adb5fca1bc1bd53883fd1dfe8f5ca824)
*   :memo: Updates documentation and removes redundant files [`b7436f3`](https://github.com/rickstaa/stable-learning-control/commit/b7436f3d1811261e4032b5d6693ebcf919c06c65)
*   :recycle: Clean up CI script [`1bab689`](https://github.com/rickstaa/stable-learning-control/commit/1bab68975e3d43c2e4cf5d9effcbd085373cf487)
*   :fire: Removes unused files and updates .gitignore [`53415a3`](https://github.com/rickstaa/stable-learning-control/commit/53415a3624b955bb1d6ed1a391a6182d1ad84bc9)
*   Update dependency cloudpickle to v1.6.0 [`2fb8c5d`](https://github.com/rickstaa/stable-learning-control/commit/2fb8c5d9dfc21aaee96a4c5049f4aca55323f50c)
*   :construction_worker: Updates remark ci action script [`11e22e1`](https://github.com/rickstaa/stable-learning-control/commit/11e22e13f38bbf931507d9cd770a15b8dc9c1190)
*   :arrow_up: Bump tensorflow from 1.15.4 to 2.4.0 in /docs [`c7cafff`](https://github.com/rickstaa/stable-learning-control/commit/c7caffff1b9161e975e102fe726ccabe8eaf77f9)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`23fb67e`](https://github.com/rickstaa/stable-learning-control/commit/23fb67e25682f157978690831f5850dc9b4f282d)
*   :twisted_rightwards_arrows: Merge branch 'add_lac_algorithm' into main [`81b88cd`](https://github.com/rickstaa/stable-learning-control/commit/81b88cdba72aa4e4d97f9ed0319fa798ca42049c)
*   :wrench: Adds dependabot configuration file [`62098ca`](https://github.com/rickstaa/stable-learning-control/commit/62098caecaa62d5b575ca19f9f0ec390cebfd66d)
*   :alien: Updates submodules [`92e6575`](https://github.com/rickstaa/stable-learning-control/commit/92e65757d9bfcda8d760c1bedf96717cbf252707)
*   :bug: Fix CI config syntax error [`b0a67ea`](https://github.com/rickstaa/stable-learning-control/commit/b0a67eaef8ffd0621d5d40cb41e8cb7f0bc5e635)
*   :bug: Adds missing system dependencies to CI [`5b981e5`](https://github.com/rickstaa/stable-learning-control/commit/5b981e5db4f779263d18e4bb74c7a9040b4bf880)
*   :bug: Always run cache CI [`49174cb`](https://github.com/rickstaa/stable-learning-control/commit/49174cbcb20daa0894ba80547b05c9ba3fa085ed)
*   :bug: Set CI failfast to false [`8445143`](https://github.com/rickstaa/stable-learning-control/commit/8445143b5e1dcd9b9f674fca0c5c613b3570024a)
*   :wrench: Updates CI config file [`7bc1c73`](https://github.com/rickstaa/stable-learning-control/commit/7bc1c73b325efad232089da27964ab8cc8880e36)
*   :bug: Fixes CI python cache id [`6553190`](https://github.com/rickstaa/stable-learning-control/commit/65531906a53415fb6947336977321776c7c60422)
*   :bug: Fixes bug in github actions [`b239fc3`](https://github.com/rickstaa/stable-learning-control/commit/b239fc310869816bddc077c0b2274f6a9f40e772)
*   :bug: Fixes bug in github actions [`533e873`](https://github.com/rickstaa/stable-learning-control/commit/533e873a9c680b1f57b49b36b959bb7b3f28c792)
*   :wrench: Updates dependabot config file [`db109ec`](https://github.com/rickstaa/stable-learning-control/commit/db109ecb51b19a6c0292e576c1bff8923553a39e)
*   :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/machine_learning_control into main [`d8f8450`](https://github.com/rickstaa/stable-learning-control/commit/d8f8450f6fff6026dfba8f15d6a9a16921b3340b)
*   :sparkles: Adds the LAC algorithm [`cd51914`](https://github.com/rickstaa/stable-learning-control/commit/cd51914e0d27b0464511037da945191e6adebbbd)
*   :sparkles: Fixes wrong learning rate decay in the SAC algorithm [`62cf02c`](https://github.com/rickstaa/stable-learning-control/commit/62cf02ca1f5ff3acea6267dd72131306874c57a5)
*   :bug: Fixes some small bugs in the LAC algorithm [`0591e07`](https://github.com/rickstaa/stable-learning-control/commit/0591e07c726f454a129dbd66fb3136cb1130e88a)
*   :sparkles: Adds SAC Algorithm [`3cb1ea4`](https://github.com/rickstaa/stable-learning-control/commit/3cb1ea4dd292608a0c05daa8272ad8dde655b9a7)
*   :sparkles: Adds working SAC implementation [`4cac2cf`](https://github.com/rickstaa/stable-learning-control/commit/4cac2cfca7bbab987b46881d3491c8641a502090)
*   :fire: Removes unused code and files [`7a9889d`](https://github.com/rickstaa/stable-learning-control/commit/7a9889d739e04a8bff0f671c27cf3592efe2f1ee)
*   :pencil: Adds documentation template [`fbafec2`](https://github.com/rickstaa/stable-learning-control/commit/fbafec25cce474a2694317f18afbe3a3409e71f8)
*   :pencil: Updates README.md and cleans up LAC and SAC code [`e76c0e8`](https://github.com/rickstaa/stable-learning-control/commit/e76c0e8e858e9e02d0c8606226d1274c4e31dee9)
*   :fire: Removes broken submodule [`6570cdc`](https://github.com/rickstaa/stable-learning-control/commit/6570cdcca21e03c73a8aef5a6421821eead621fa)
*   :sparkles: Changes the SAC learning rate decay from step to episode based [`08fea9a`](https://github.com/rickstaa/stable-learning-control/commit/08fea9a83addf2c5987367e6528ae1cf090690af)
*   :wrench: Updates CI config [`6289ae3`](https://github.com/rickstaa/stable-learning-control/commit/6289ae392c732f7443387d14c9b71248c4e9c326)
*   :art: Improves code readability and pulls submodules [`4efa66b`](https://github.com/rickstaa/stable-learning-control/commit/4efa66b8df1c8816c856d95474d6899f49f6a17c)
*   :sparkles: Adds multiple python versions tests to CI [`616e592`](https://github.com/rickstaa/stable-learning-control/commit/616e592a07140129a0f9624de3436d87f88b20f1)
*   :bug: Fixes increasing lambda error [`ede5d6b`](https://github.com/rickstaa/stable-learning-control/commit/ede5d6bf3beeb04874621d1888a6378579d2ff14)
*   :art: Improves code formatting [`5db397a`](https://github.com/rickstaa/stable-learning-control/commit/5db397ae0f281ff1171091a3a80733e752042873)
*   :pencil: Updates README, removes unused files and pulls submodules [`9766020`](https://github.com/rickstaa/stable-learning-control/commit/9766020c486e4691ffae1882ad7514e1f7675905)
*   :bug: Fixes CI cache bug [`afb146d`](https://github.com/rickstaa/stable-learning-control/commit/afb146d6372a3a68410deddaf43a67e18126a268)
*   :bug: Changes flak8 action [`e4b549a`](https://github.com/rickstaa/stable-learning-control/commit/e4b549a2be8a8dda5b96097dbfe5dac828a1b460)
*   :bug: Changes flake8 CI action provider [`1237885`](https://github.com/rickstaa/stable-learning-control/commit/1237885d402a3c6b2f6fb3aebea0fe9c684e88d7)
*   :bug: Fixes the algorithm run scripts to have the right arguments [`55ef144`](https://github.com/rickstaa/stable-learning-control/commit/55ef14479f9e1ddb30c170d97be05ce68720330b)
*   :bug: Fixes flake 8 CI command error [`4362067`](https://github.com/rickstaa/stable-learning-control/commit/4362067712d04821abd1f153e97ae16733016343)
*   :bug: Changes flak8 anotator CI [`945083d`](https://github.com/rickstaa/stable-learning-control/commit/945083d754598d70c2547bdf177fdb13af498525)
*   :sparkles: Updates simzoo submodule [`4a4b73b`](https://github.com/rickstaa/stable-learning-control/commit/4a4b73b6ade6f64ba1d00eeb8995151a3bd1a02e)
*   :sparkles: Adds simzoo submodule [`a87c848`](https://github.com/rickstaa/stable-learning-control/commit/a87c848001b95773cf1255861cddc7c4e8b9cafe)
*   :bug: Updates CI script formatting [`c06fc75`](https://github.com/rickstaa/stable-learning-control/commit/c06fc7588292c33f3c7361c7c9403fb6ec0a2a7c)
*   :bug: Fixes pytest artifacts upload bug CI [`5b73136`](https://github.com/rickstaa/stable-learning-control/commit/5b73136eaed361e41b1cd3d9f0fb0566c41194f1)
*   :bug: Fixes CI format bug [`25671b6`](https://github.com/rickstaa/stable-learning-control/commit/25671b6e4d1c9b15d0da4c6aa590ed8a8c4960e7)
*   :bug: Fixes CI python cache problem [`0ac5fa9`](https://github.com/rickstaa/stable-learning-control/commit/0ac5fa9a2174257b412240f03e099c87b4ad96dc)
*   :bug: Fix CI syntax error [`dd8356b`](https://github.com/rickstaa/stable-learning-control/commit/dd8356b522ee9ec1f1038c8ccc06140412ede989)
*   :bug: Fixes remark link CI config [`ff9005a`](https://github.com/rickstaa/stable-learning-control/commit/ff9005a1a47848abe82eba6dc9ec59eddea70cbe)
*   :bug; Fix CI cache operation [`51c126e`](https://github.com/rickstaa/stable-learning-control/commit/51c126ed33c2dacbfa61333a7ce114e593a08d02)
*   :bug: Updates markdown CI config [`7cecd82`](https://github.com/rickstaa/stable-learning-control/commit/7cecd820b502c91215ca2d4a7085caa2dbd95adb)
*   :bug: Fixes pylint bug [`540bd53`](https://github.com/rickstaa/stable-learning-control/commit/540bd53b2ae929b784a6d1033f88291589285050)
*   :bug: Fixes CI format bug [`5018c9e`](https://github.com/rickstaa/stable-learning-control/commit/5018c9e0a7d1dae864bf2dc0e746245aca18582f)
*   :bug: Fix CI config syntax bug [`bd760f0`](https://github.com/rickstaa/stable-learning-control/commit/bd760f056da9568772a53a55ce12e7f34534fa0c)
*   :bug: Fixes dev python dependencies [`3733f18`](https://github.com/rickstaa/stable-learning-control/commit/3733f18ef13c04980acc5f2c4ae0793c5929a60b)
*   :bug; Fix CI cache operation [`dcffef6`](https://github.com/rickstaa/stable-learning-control/commit/dcffef632a7e89fd503cf8a423121638b4ec9156)
*   :bug: Fix CI syntax error [`b322543`](https://github.com/rickstaa/stable-learning-control/commit/b32254366b3ad9131a7bf47e56f683f014924852)
*   :bug: Fixes CI black files bug [`add3364`](https://github.com/rickstaa/stable-learning-control/commit/add3364821c2a29e1710690f6848f2d3bd8e949c)
*   :twisted_rightwards_arrows: Merges branch add_lac_algorithm into master [`7457680`](https://github.com/rickstaa/stable-learning-control/commit/74576805264f24f86b3ce51027d8750b7bbfd3f5)
*   :sparkles: Updates environments [`829bf94`](https://github.com/rickstaa/stable-learning-control/commit/829bf94f4965e0b52b7116e81d0684597ca7f27c)
*   :sparkles: Updates simzoo submodule [`bfb75ea`](https://github.com/rickstaa/stable-learning-control/commit/bfb75eaf76435f918372d7299847f0dc8225d727)

## v0.0.0 - 2020-08-14

### Merged

*   :bug: Fixes bug in the oscillator environment [`#7`](https://github.com/rickstaa/stable-learning-control/pull/7)
*   :sparkles: Adds oscillator environment [`#4`](https://github.com/rickstaa/stable-learning-control/pull/4)

### Commits

*   :building_construction: Changes package structure [`bbcd737`](https://github.com/rickstaa/stable-learning-control/commit/bbcd73760c44ea50419db1fcaf9bae4704d774a2)
*   :twisted_rightwards_arrows: Merge branch 'adds_gym_env_registration' into add_lac_algorithm [`beee921`](https://github.com/rickstaa/stable-learning-control/commit/beee9215f542a402e4b4f71c9268fe276ac9a530)
*   :fire: Removes old LAC algorithm files [`25a0a5f`](https://github.com/rickstaa/stable-learning-control/commit/25a0a5f0e3a4b8ef52567617bb65ef1114ba60ea)
*   :wastebasket: Adds original LAC code to the repository [`c4c4587`](https://github.com/rickstaa/stable-learning-control/commit/c4c458731b75a65b3e63d0879e43860956d06c5a)
*   :see_no_evil: Updates gitignore [`3626d92`](https://github.com/rickstaa/stable-learning-control/commit/3626d92e6f3d4fb720979d59ca403bdf18c50114)
*   :truck: Changes simzoo folder structure and add setup.py [`510da5b`](https://github.com/rickstaa/stable-learning-control/commit/510da5b266ef07133e2914146a8a5cbdf98a6eb9)
*   :bug: Fixes some bugs in the oscillator environment [`edb1c2e`](https://github.com/rickstaa/stable-learning-control/commit/edb1c2e4765ced89a7bf0b4e4423705c921625f2)
*   :sparkles: Creates user story and bug issue template [`c45cbf3`](https://github.com/rickstaa/stable-learning-control/commit/c45cbf3d35c6fa6185e8d83bded0b520b9b11816)
*   :building_construction: Updates folder structure [`cb5d28a`](https://github.com/rickstaa/stable-learning-control/commit/cb5d28a9aace73dcb4c643f8ca2d55052b94e4b2)
*   :building_construction: Updates folder structure [`50e17b2`](https://github.com/rickstaa/stable-learning-control/commit/50e17b2c99a46d16da921b8e285a9852ca042f11)
*   :white_check_mark: Updates python tests and gitignore [`3213d1c`](https://github.com/rickstaa/stable-learning-control/commit/3213d1cdbeaf11873406d50c258e9677a44d2e84)
*   :twisted_rightwards_arrows: Merge branch 'adds_gym_env_registration' into add_lac_algorithm [`67006a2`](https://github.com/rickstaa/stable-learning-control/commit/67006a27b359970af1db23675c9ca8402413233d)
*   :sparkles: Adds gym environment registration [`eed9a8a`](https://github.com/rickstaa/stable-learning-control/commit/eed9a8ab815171d470369e05a2a10a5af5e704ee)
*   :see_no_evil: Adds gitignore [`dd67102`](https://github.com/rickstaa/stable-learning-control/commit/dd671028361d5bbd3831a11679eda8bd8f194245)
*   :pencil: Updates README.md [`de683d4`](https://github.com/rickstaa/stable-learning-control/commit/de683d42c5333ce62528661c64c9a6469c54392b)
*   :art: Updates user story issue template [`78b52cd`](https://github.com/rickstaa/stable-learning-control/commit/78b52cd4df1243638d2957202ccd7639eb895bfb)
*   :fire: Removes python cache files [`60ef3b4`](https://github.com/rickstaa/stable-learning-control/commit/60ef3b4e10cdb7e4fd6add7f32e14e84e965717a)
*   :twisted_rightwards_arrows: Merges branch 'master' into add_lac_algorithm [`ac480a2`](https://github.com/rickstaa/stable-learning-control/commit/ac480a23b3d3ad0114603d61d5f7dcd663999aa2)
*   :twisted_rightwards_arrows: Merge branch 'change_simzoo_structure2' into add_lac_algorithm [`3ab9e15`](https://github.com/rickstaa/stable-learning-control/commit/3ab9e15789775fc6a334d58ef67e86eb44309820)
*   :tada: First commit [`fedf77c`](https://github.com/rickstaa/stable-learning-control/commit/fedf77c88519de911cb3597af47519bdb1ffd835)
