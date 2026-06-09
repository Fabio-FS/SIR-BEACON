```
# Project Structure

├── Final.afdesign
├── SI.afdesign
├── SI.afdesign~lock~
├── backend
│   ├── main.py
│   ├── requirements.txt
│   └── simulation.py
├── envirorment.yml
├── frontend
│   ├── index.html
│   ├── node_modules
│   │   ├── .bin
│   │   │   ├── baseline-browser-mapping
│   │   │   ├── baseline-browser-mapping.cmd
│   │   │   ├── baseline-browser-mapping.ps1
│   │   │   ├── browserslist
│   │   │   ├── browserslist.cmd
│   │   │   ├── browserslist.ps1
│   │   │   ├── jsesc
│   │   │   ├── jsesc.cmd
│   │   │   ├── jsesc.ps1
│   │   │   ├── json5
│   │   │   ├── json5.cmd
│   │   │   ├── json5.ps1
│   │   │   ├── loose-envify
│   │   │   ├── loose-envify.cmd
│   │   │   ├── loose-envify.ps1
│   │   │   ├── nanoid
│   │   │   ├── nanoid.cmd
│   │   │   ├── nanoid.ps1
│   │   │   ├── parser
│   │   │   ├── parser.cmd
│   │   │   ├── parser.ps1
│   │   │   ├── rollup
│   │   │   ├── rollup.cmd
│   │   │   ├── rollup.ps1
│   │   │   ├── semver
│   │   │   ├── semver.cmd
│   │   │   ├── semver.ps1
│   │   │   ├── update-browserslist-db
│   │   │   ├── update-browserslist-db.cmd
│   │   │   ├── update-browserslist-db.ps1
│   │   │   ├── vite
│   │   │   ├── vite.cmd
│   │   │   └── vite.ps1
│   │   ├── .package-lock.json
│   │   ├── .vite
│   │   │   └── deps
│   │   │       ├── _metadata.json
│   │   │       ├── chunk-373CG7ZK.js
│   │   │       ├── chunk-373CG7ZK.js.map
│   │   │       ├── chunk-REFQX4J5.js
│   │   │       ├── chunk-REFQX4J5.js.map
│   │   │       ├── package.json
│   │   │       ├── react-dom.js
│   │   │       ├── react-dom.js.map
│   │   │       ├── react-dom_client.js
│   │   │       ├── react-dom_client.js.map
│   │   │       ├── react.js
│   │   │       ├── react.js.map
│   │   │       ├── react_jsx-dev-runtime.js
│   │   │       ├── react_jsx-dev-runtime.js.map
│   │   │       ├── react_jsx-runtime.js
│   │   │       └── react_jsx-runtime.js.map
│   │   ├── @babel
│   │   │   ├── code-frame
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── compat-data
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── corejs2-built-ins.js
│   │   │   │   ├── corejs3-shipped-proposals.js
│   │   │   │   ├── data
│   │   │   │   │   ├── corejs2-built-ins.json
│   │   │   │   │   ├── corejs3-shipped-proposals.json
│   │   │   │   │   ├── native-modules.json
│   │   │   │   │   ├── overlapping-plugins.json
│   │   │   │   │   ├── plugin-bugfixes.json
│   │   │   │   │   └── plugins.json
│   │   │   │   ├── native-modules.js
│   │   │   │   ├── overlapping-plugins.js
│   │   │   │   ├── package.json
│   │   │   │   ├── plugin-bugfixes.js
│   │   │   │   └── plugins.js
│   │   │   ├── core
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── config
│   │   │   │   │   │   ├── cache-contexts.js
│   │   │   │   │   │   ├── cache-contexts.js.map
│   │   │   │   │   │   ├── caching.js
│   │   │   │   │   │   ├── caching.js.map
│   │   │   │   │   │   ├── config-chain.js
│   │   │   │   │   │   ├── config-chain.js.map
│   │   │   │   │   │   ├── config-descriptors.js
│   │   │   │   │   │   ├── config-descriptors.js.map
│   │   │   │   │   │   ├── files
│   │   │   │   │   │   │   ├── configuration.js
│   │   │   │   │   │   │   ├── configuration.js.map
│   │   │   │   │   │   │   ├── import.cjs
│   │   │   │   │   │   │   ├── import.cjs.map
│   │   │   │   │   │   │   ├── index-browser.js
│   │   │   │   │   │   │   ├── index-browser.js.map
│   │   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   │   ├── module-types.js
│   │   │   │   │   │   │   ├── module-types.js.map
│   │   │   │   │   │   │   ├── package.js
│   │   │   │   │   │   │   ├── package.js.map
│   │   │   │   │   │   │   ├── plugins.js
│   │   │   │   │   │   │   ├── plugins.js.map
│   │   │   │   │   │   │   ├── types.js
│   │   │   │   │   │   │   ├── types.js.map
│   │   │   │   │   │   │   ├── utils.js
│   │   │   │   │   │   │   └── utils.js.map
│   │   │   │   │   │   ├── full.js
│   │   │   │   │   │   ├── full.js.map
│   │   │   │   │   │   ├── helpers
│   │   │   │   │   │   │   ├── config-api.js
│   │   │   │   │   │   │   ├── config-api.js.map
│   │   │   │   │   │   │   ├── deep-array.js
│   │   │   │   │   │   │   ├── deep-array.js.map
│   │   │   │   │   │   │   ├── environment.js
│   │   │   │   │   │   │   └── environment.js.map
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── item.js
│   │   │   │   │   │   ├── item.js.map
│   │   │   │   │   │   ├── partial.js
│   │   │   │   │   │   ├── partial.js.map
│   │   │   │   │   │   ├── pattern-to-regex.js
│   │   │   │   │   │   ├── pattern-to-regex.js.map
│   │   │   │   │   │   ├── plugin.js
│   │   │   │   │   │   ├── plugin.js.map
│   │   │   │   │   │   ├── printer.js
│   │   │   │   │   │   ├── printer.js.map
│   │   │   │   │   │   ├── resolve-targets-browser.js
│   │   │   │   │   │   ├── resolve-targets-browser.js.map
│   │   │   │   │   │   ├── resolve-targets.js
│   │   │   │   │   │   ├── resolve-targets.js.map
│   │   │   │   │   │   ├── util.js
│   │   │   │   │   │   ├── util.js.map
│   │   │   │   │   │   └── validation
│   │   │   │   │   │       ├── option-assertions.js
│   │   │   │   │   │       ├── option-assertions.js.map
│   │   │   │   │   │       ├── options.js
│   │   │   │   │   │       ├── options.js.map
│   │   │   │   │   │       ├── plugins.js
│   │   │   │   │   │       ├── plugins.js.map
│   │   │   │   │   │       ├── removed.js
│   │   │   │   │   │       └── removed.js.map
│   │   │   │   │   ├── errors
│   │   │   │   │   │   ├── config-error.js
│   │   │   │   │   │   ├── config-error.js.map
│   │   │   │   │   │   ├── rewrite-stack-trace.js
│   │   │   │   │   │   └── rewrite-stack-trace.js.map
│   │   │   │   │   ├── gensync-utils
│   │   │   │   │   │   ├── async.js
│   │   │   │   │   │   ├── async.js.map
│   │   │   │   │   │   ├── fs.js
│   │   │   │   │   │   ├── fs.js.map
│   │   │   │   │   │   ├── functional.js
│   │   │   │   │   │   └── functional.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── parse.js
│   │   │   │   │   ├── parse.js.map
│   │   │   │   │   ├── parser
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   └── util
│   │   │   │   │   │       ├── missing-plugin-helper.js
│   │   │   │   │   │       └── missing-plugin-helper.js.map
│   │   │   │   │   ├── tools
│   │   │   │   │   ├── transform-ast.js
│   │   │   │   │   ├── transform-ast.js.map
│   │   │   │   │   ├── transform-file-browser.js
│   │   │   │   │   ├── transform-file-browser.js.map
│   │   │   │   │   ├── transform-file.js
│   │   │   │   │   ├── transform-file.js.map
│   │   │   │   │   ├── transform.js
│   │   │   │   │   ├── transform.js.map
│   │   │   │   │   ├── transformation
│   │   │   │   │   │   ├── block-hoist-plugin.js
│   │   │   │   │   │   ├── block-hoist-plugin.js.map
│   │   │   │   │   │   ├── file
│   │   │   │   │   │   │   ├── babel-7-helpers.cjs
│   │   │   │   │   │   │   ├── babel-7-helpers.cjs.map
│   │   │   │   │   │   │   ├── file.js
│   │   │   │   │   │   │   ├── file.js.map
│   │   │   │   │   │   │   ├── generate.js
│   │   │   │   │   │   │   ├── generate.js.map
│   │   │   │   │   │   │   ├── merge-map.js
│   │   │   │   │   │   │   └── merge-map.js.map
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── normalize-file.js
│   │   │   │   │   │   ├── normalize-file.js.map
│   │   │   │   │   │   ├── normalize-opts.js
│   │   │   │   │   │   ├── normalize-opts.js.map
│   │   │   │   │   │   ├── plugin-pass.js
│   │   │   │   │   │   ├── plugin-pass.js.map
│   │   │   │   │   │   ├── read-input-source-map-file-browser.js
│   │   │   │   │   │   ├── read-input-source-map-file-browser.js.map
│   │   │   │   │   │   ├── read-input-source-map-file.js
│   │   │   │   │   │   ├── read-input-source-map-file.js.map
│   │   │   │   │   │   └── util
│   │   │   │   │   │       ├── clone-deep.js
│   │   │   │   │   │       └── clone-deep.js.map
│   │   │   │   │   └── vendor
│   │   │   │   │       ├── import-meta-resolve.js
│   │   │   │   │       └── import-meta-resolve.js.map
│   │   │   │   ├── package.json
│   │   │   │   └── src
│   │   │   │       ├── config
│   │   │   │       │   ├── files
│   │   │   │       │   │   ├── index-browser.ts
│   │   │   │       │   │   └── index.ts
│   │   │   │       │   ├── resolve-targets-browser.ts
│   │   │   │       │   └── resolve-targets.ts
│   │   │   │       ├── transform-file-browser.ts
│   │   │   │       ├── transform-file.ts
│   │   │   │       └── transformation
│   │   │   │           ├── read-input-source-map-file-browser.ts
│   │   │   │           └── read-input-source-map-file.ts
│   │   │   ├── generator
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── buffer.js
│   │   │   │   │   ├── buffer.js.map
│   │   │   │   │   ├── generators
│   │   │   │   │   │   ├── base.js
│   │   │   │   │   │   ├── base.js.map
│   │   │   │   │   │   ├── classes.js
│   │   │   │   │   │   ├── classes.js.map
│   │   │   │   │   │   ├── deprecated.js
│   │   │   │   │   │   ├── deprecated.js.map
│   │   │   │   │   │   ├── expressions.js
│   │   │   │   │   │   ├── expressions.js.map
│   │   │   │   │   │   ├── flow.js
│   │   │   │   │   │   ├── flow.js.map
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── jsx.js
│   │   │   │   │   │   ├── jsx.js.map
│   │   │   │   │   │   ├── methods.js
│   │   │   │   │   │   ├── methods.js.map
│   │   │   │   │   │   ├── modules.js
│   │   │   │   │   │   ├── modules.js.map
│   │   │   │   │   │   ├── statements.js
│   │   │   │   │   │   ├── statements.js.map
│   │   │   │   │   │   ├── template-literals.js
│   │   │   │   │   │   ├── template-literals.js.map
│   │   │   │   │   │   ├── types.js
│   │   │   │   │   │   ├── types.js.map
│   │   │   │   │   │   ├── typescript.js
│   │   │   │   │   │   └── typescript.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── node
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── parentheses.js
│   │   │   │   │   │   └── parentheses.js.map
│   │   │   │   │   ├── nodes.js
│   │   │   │   │   ├── nodes.js.map
│   │   │   │   │   ├── printer.js
│   │   │   │   │   ├── printer.js.map
│   │   │   │   │   ├── source-map.js
│   │   │   │   │   ├── source-map.js.map
│   │   │   │   │   ├── token-map.js
│   │   │   │   │   └── token-map.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-compilation-targets
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── debug.js
│   │   │   │   │   ├── debug.js.map
│   │   │   │   │   ├── filter-items.js
│   │   │   │   │   ├── filter-items.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── options.js
│   │   │   │   │   ├── options.js.map
│   │   │   │   │   ├── pretty.js
│   │   │   │   │   ├── pretty.js.map
│   │   │   │   │   ├── targets.js
│   │   │   │   │   ├── targets.js.map
│   │   │   │   │   ├── utils.js
│   │   │   │   │   └── utils.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-globals
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── data
│   │   │   │   │   ├── browser-upper.json
│   │   │   │   │   ├── builtin-lower.json
│   │   │   │   │   └── builtin-upper.json
│   │   │   │   └── package.json
│   │   │   ├── helper-module-imports
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── import-injector.js
│   │   │   │   │   ├── import-injector.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── is-module.js
│   │   │   │   │   └── is-module.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-module-transforms
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── dynamic-import.js
│   │   │   │   │   ├── dynamic-import.js.map
│   │   │   │   │   ├── get-module-name.js
│   │   │   │   │   ├── get-module-name.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── lazy-modules.js
│   │   │   │   │   ├── lazy-modules.js.map
│   │   │   │   │   ├── normalize-and-load-metadata.js
│   │   │   │   │   ├── normalize-and-load-metadata.js.map
│   │   │   │   │   ├── rewrite-live-references.js
│   │   │   │   │   ├── rewrite-live-references.js.map
│   │   │   │   │   ├── rewrite-this.js
│   │   │   │   │   └── rewrite-this.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-plugin-utils
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-string-parser
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-validator-identifier
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── identifier.js
│   │   │   │   │   ├── identifier.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── keyword.js
│   │   │   │   │   └── keyword.js.map
│   │   │   │   └── package.json
│   │   │   ├── helper-validator-option
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── find-suggestion.js
│   │   │   │   │   ├── find-suggestion.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── validator.js
│   │   │   │   │   └── validator.js.map
│   │   │   │   └── package.json
│   │   │   ├── helpers
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── helpers
│   │   │   │   │   │   ├── AwaitValue.js
│   │   │   │   │   │   ├── AwaitValue.js.map
│   │   │   │   │   │   ├── OverloadYield.js
│   │   │   │   │   │   ├── OverloadYield.js.map
│   │   │   │   │   │   ├── applyDecoratedDescriptor.js
│   │   │   │   │   │   ├── applyDecoratedDescriptor.js.map
│   │   │   │   │   │   ├── applyDecs.js
│   │   │   │   │   │   ├── applyDecs.js.map
│   │   │   │   │   │   ├── applyDecs2203.js
│   │   │   │   │   │   ├── applyDecs2203.js.map
│   │   │   │   │   │   ├── applyDecs2203R.js
│   │   │   │   │   │   ├── applyDecs2203R.js.map
│   │   │   │   │   │   ├── applyDecs2301.js
│   │   │   │   │   │   ├── applyDecs2301.js.map
│   │   │   │   │   │   ├── applyDecs2305.js
│   │   │   │   │   │   ├── applyDecs2305.js.map
│   │   │   │   │   │   ├── applyDecs2311.js
│   │   │   │   │   │   ├── applyDecs2311.js.map
│   │   │   │   │   │   ├── arrayLikeToArray.js
│   │   │   │   │   │   ├── arrayLikeToArray.js.map
│   │   │   │   │   │   ├── arrayWithHoles.js
│   │   │   │   │   │   ├── arrayWithHoles.js.map
│   │   │   │   │   │   ├── arrayWithoutHoles.js
│   │   │   │   │   │   ├── arrayWithoutHoles.js.map
│   │   │   │   │   │   ├── assertClassBrand.js
│   │   │   │   │   │   ├── assertClassBrand.js.map
│   │   │   │   │   │   ├── assertThisInitialized.js
│   │   │   │   │   │   ├── assertThisInitialized.js.map
│   │   │   │   │   │   ├── asyncGeneratorDelegate.js
│   │   │   │   │   │   ├── asyncGeneratorDelegate.js.map
│   │   │   │   │   │   ├── asyncIterator.js
│   │   │   │   │   │   ├── asyncIterator.js.map
│   │   │   │   │   │   ├── asyncToGenerator.js
│   │   │   │   │   │   ├── asyncToGenerator.js.map
│   │   │   │   │   │   ├── awaitAsyncGenerator.js
│   │   │   │   │   │   ├── awaitAsyncGenerator.js.map
│   │   │   │   │   │   ├── callSuper.js
│   │   │   │   │   │   ├── callSuper.js.map
│   │   │   │   │   │   ├── checkInRHS.js
│   │   │   │   │   │   ├── checkInRHS.js.map
│   │   │   │   │   │   ├── checkPrivateRedeclaration.js
│   │   │   │   │   │   ├── checkPrivateRedeclaration.js.map
│   │   │   │   │   │   ├── classApplyDescriptorDestructureSet.js
│   │   │   │   │   │   ├── classApplyDescriptorDestructureSet.js.map
│   │   │   │   │   │   ├── classApplyDescriptorGet.js
│   │   │   │   │   │   ├── classApplyDescriptorGet.js.map
│   │   │   │   │   │   ├── classApplyDescriptorSet.js
│   │   │   │   │   │   ├── classApplyDescriptorSet.js.map
│   │   │   │   │   │   ├── classCallCheck.js
│   │   │   │   │   │   ├── classCallCheck.js.map
│   │   │   │   │   │   ├── classCheckPrivateStaticAccess.js
│   │   │   │   │   │   ├── classCheckPrivateStaticAccess.js.map
│   │   │   │   │   │   ├── classCheckPrivateStaticFieldDescriptor.js
│   │   │   │   │   │   ├── classCheckPrivateStaticFieldDescriptor.js.map
│   │   │   │   │   │   ├── classExtractFieldDescriptor.js
│   │   │   │   │   │   ├── classExtractFieldDescriptor.js.map
│   │   │   │   │   │   ├── classNameTDZError.js
│   │   │   │   │   │   ├── classNameTDZError.js.map
│   │   │   │   │   │   ├── classPrivateFieldDestructureSet.js
│   │   │   │   │   │   ├── classPrivateFieldDestructureSet.js.map
│   │   │   │   │   │   ├── classPrivateFieldGet.js
│   │   │   │   │   │   ├── classPrivateFieldGet.js.map
│   │   │   │   │   │   ├── classPrivateFieldGet2.js
│   │   │   │   │   │   ├── classPrivateFieldGet2.js.map
│   │   │   │   │   │   ├── classPrivateFieldInitSpec.js
│   │   │   │   │   │   ├── classPrivateFieldInitSpec.js.map
│   │   │   │   │   │   ├── classPrivateFieldLooseBase.js
│   │   │   │   │   │   ├── classPrivateFieldLooseBase.js.map
│   │   │   │   │   │   ├── classPrivateFieldLooseKey.js
│   │   │   │   │   │   ├── classPrivateFieldLooseKey.js.map
│   │   │   │   │   │   ├── classPrivateFieldSet.js
│   │   │   │   │   │   ├── classPrivateFieldSet.js.map
│   │   │   │   │   │   ├── classPrivateFieldSet2.js
│   │   │   │   │   │   ├── classPrivateFieldSet2.js.map
│   │   │   │   │   │   ├── classPrivateGetter.js
│   │   │   │   │   │   ├── classPrivateGetter.js.map
│   │   │   │   │   │   ├── classPrivateMethodGet.js
│   │   │   │   │   │   ├── classPrivateMethodGet.js.map
│   │   │   │   │   │   ├── classPrivateMethodInitSpec.js
│   │   │   │   │   │   ├── classPrivateMethodInitSpec.js.map
│   │   │   │   │   │   ├── classPrivateMethodSet.js
│   │   │   │   │   │   ├── classPrivateMethodSet.js.map
│   │   │   │   │   │   ├── classPrivateSetter.js
│   │   │   │   │   │   ├── classPrivateSetter.js.map
│   │   │   │   │   │   ├── classStaticPrivateFieldDestructureSet.js
│   │   │   │   │   │   ├── classStaticPrivateFieldDestructureSet.js.map
│   │   │   │   │   │   ├── classStaticPrivateFieldSpecGet.js
│   │   │   │   │   │   ├── classStaticPrivateFieldSpecGet.js.map
│   │   │   │   │   │   ├── classStaticPrivateFieldSpecSet.js
│   │   │   │   │   │   ├── classStaticPrivateFieldSpecSet.js.map
│   │   │   │   │   │   ├── classStaticPrivateMethodGet.js
│   │   │   │   │   │   ├── classStaticPrivateMethodGet.js.map
│   │   │   │   │   │   ├── classStaticPrivateMethodSet.js
│   │   │   │   │   │   ├── classStaticPrivateMethodSet.js.map
│   │   │   │   │   │   ├── construct.js
│   │   │   │   │   │   ├── construct.js.map
│   │   │   │   │   │   ├── createClass.js
│   │   │   │   │   │   ├── createClass.js.map
│   │   │   │   │   │   ├── createForOfIteratorHelper.js
│   │   │   │   │   │   ├── createForOfIteratorHelper.js.map
│   │   │   │   │   │   ├── createForOfIteratorHelperLoose.js
│   │   │   │   │   │   ├── createForOfIteratorHelperLoose.js.map
│   │   │   │   │   │   ├── createSuper.js
│   │   │   │   │   │   ├── createSuper.js.map
│   │   │   │   │   │   ├── decorate.js
│   │   │   │   │   │   ├── decorate.js.map
│   │   │   │   │   │   ├── defaults.js
│   │   │   │   │   │   ├── defaults.js.map
│   │   │   │   │   │   ├── defineAccessor.js
│   │   │   │   │   │   ├── defineAccessor.js.map
│   │   │   │   │   │   ├── defineEnumerableProperties.js
│   │   │   │   │   │   ├── defineEnumerableProperties.js.map
│   │   │   │   │   │   ├── defineProperty.js
│   │   │   │   │   │   ├── defineProperty.js.map
│   │   │   │   │   │   ├── dispose.js
│   │   │   │   │   │   ├── dispose.js.map
│   │   │   │   │   │   ├── extends.js
│   │   │   │   │   │   ├── extends.js.map
│   │   │   │   │   │   ├── get.js
│   │   │   │   │   │   ├── get.js.map
│   │   │   │   │   │   ├── getPrototypeOf.js
│   │   │   │   │   │   ├── getPrototypeOf.js.map
│   │   │   │   │   │   ├── identity.js
│   │   │   │   │   │   ├── identity.js.map
│   │   │   │   │   │   ├── importDeferProxy.js
│   │   │   │   │   │   ├── importDeferProxy.js.map
│   │   │   │   │   │   ├── inherits.js
│   │   │   │   │   │   ├── inherits.js.map
│   │   │   │   │   │   ├── inheritsLoose.js
│   │   │   │   │   │   ├── inheritsLoose.js.map
│   │   │   │   │   │   ├── initializerDefineProperty.js
│   │   │   │   │   │   ├── initializerDefineProperty.js.map
│   │   │   │   │   │   ├── initializerWarningHelper.js
│   │   │   │   │   │   ├── initializerWarningHelper.js.map
│   │   │   │   │   │   ├── instanceof.js
│   │   │   │   │   │   ├── instanceof.js.map
│   │   │   │   │   │   ├── interopRequireDefault.js
│   │   │   │   │   │   ├── interopRequireDefault.js.map
│   │   │   │   │   │   ├── interopRequireWildcard.js
│   │   │   │   │   │   ├── interopRequireWildcard.js.map
│   │   │   │   │   │   ├── isNativeFunction.js
│   │   │   │   │   │   ├── isNativeFunction.js.map
│   │   │   │   │   │   ├── isNativeReflectConstruct.js
│   │   │   │   │   │   ├── isNativeReflectConstruct.js.map
│   │   │   │   │   │   ├── iterableToArray.js
│   │   │   │   │   │   ├── iterableToArray.js.map
│   │   │   │   │   │   ├── iterableToArrayLimit.js
│   │   │   │   │   │   ├── iterableToArrayLimit.js.map
│   │   │   │   │   │   ├── jsx.js
│   │   │   │   │   │   ├── jsx.js.map
│   │   │   │   │   │   ├── maybeArrayLike.js
│   │   │   │   │   │   ├── maybeArrayLike.js.map
│   │   │   │   │   │   ├── newArrowCheck.js
│   │   │   │   │   │   ├── newArrowCheck.js.map
│   │   │   │   │   │   ├── nonIterableRest.js
│   │   │   │   │   │   ├── nonIterableRest.js.map
│   │   │   │   │   │   ├── nonIterableSpread.js
│   │   │   │   │   │   ├── nonIterableSpread.js.map
│   │   │   │   │   │   ├── nullishReceiverError.js
│   │   │   │   │   │   ├── nullishReceiverError.js.map
│   │   │   │   │   │   ├── objectDestructuringEmpty.js
│   │   │   │   │   │   ├── objectDestructuringEmpty.js.map
│   │   │   │   │   │   ├── objectSpread.js
│   │   │   │   │   │   ├── objectSpread.js.map
│   │   │   │   │   │   ├── objectSpread2.js
│   │   │   │   │   │   ├── objectSpread2.js.map
│   │   │   │   │   │   ├── objectWithoutProperties.js
│   │   │   │   │   │   ├── objectWithoutProperties.js.map
│   │   │   │   │   │   ├── objectWithoutPropertiesLoose.js
│   │   │   │   │   │   ├── objectWithoutPropertiesLoose.js.map
│   │   │   │   │   │   ├── possibleConstructorReturn.js
│   │   │   │   │   │   ├── possibleConstructorReturn.js.map
│   │   │   │   │   │   ├── readOnlyError.js
│   │   │   │   │   │   ├── readOnlyError.js.map
│   │   │   │   │   │   ├── regenerator.js
│   │   │   │   │   │   ├── regenerator.js.map
│   │   │   │   │   │   ├── regeneratorAsync.js
│   │   │   │   │   │   ├── regeneratorAsync.js.map
│   │   │   │   │   │   ├── regeneratorAsyncGen.js
│   │   │   │   │   │   ├── regeneratorAsyncGen.js.map
│   │   │   │   │   │   ├── regeneratorAsyncIterator.js
│   │   │   │   │   │   ├── regeneratorAsyncIterator.js.map
│   │   │   │   │   │   ├── regeneratorDefine.js
│   │   │   │   │   │   ├── regeneratorDefine.js.map
│   │   │   │   │   │   ├── regeneratorKeys.js
│   │   │   │   │   │   ├── regeneratorKeys.js.map
│   │   │   │   │   │   ├── regeneratorRuntime.js
│   │   │   │   │   │   ├── regeneratorRuntime.js.map
│   │   │   │   │   │   ├── regeneratorValues.js
│   │   │   │   │   │   ├── regeneratorValues.js.map
│   │   │   │   │   │   ├── set.js
│   │   │   │   │   │   ├── set.js.map
│   │   │   │   │   │   ├── setFunctionName.js
│   │   │   │   │   │   ├── setFunctionName.js.map
│   │   │   │   │   │   ├── setPrototypeOf.js
│   │   │   │   │   │   ├── setPrototypeOf.js.map
│   │   │   │   │   │   ├── skipFirstGeneratorNext.js
│   │   │   │   │   │   ├── skipFirstGeneratorNext.js.map
│   │   │   │   │   │   ├── slicedToArray.js
│   │   │   │   │   │   ├── slicedToArray.js.map
│   │   │   │   │   │   ├── superPropBase.js
│   │   │   │   │   │   ├── superPropBase.js.map
│   │   │   │   │   │   ├── superPropGet.js
│   │   │   │   │   │   ├── superPropGet.js.map
│   │   │   │   │   │   ├── superPropSet.js
│   │   │   │   │   │   ├── superPropSet.js.map
│   │   │   │   │   │   ├── taggedTemplateLiteral.js
│   │   │   │   │   │   ├── taggedTemplateLiteral.js.map
│   │   │   │   │   │   ├── taggedTemplateLiteralLoose.js
│   │   │   │   │   │   ├── taggedTemplateLiteralLoose.js.map
│   │   │   │   │   │   ├── tdz.js
│   │   │   │   │   │   ├── tdz.js.map
│   │   │   │   │   │   ├── temporalRef.js
│   │   │   │   │   │   ├── temporalRef.js.map
│   │   │   │   │   │   ├── temporalUndefined.js
│   │   │   │   │   │   ├── temporalUndefined.js.map
│   │   │   │   │   │   ├── toArray.js
│   │   │   │   │   │   ├── toArray.js.map
│   │   │   │   │   │   ├── toConsumableArray.js
│   │   │   │   │   │   ├── toConsumableArray.js.map
│   │   │   │   │   │   ├── toPrimitive.js
│   │   │   │   │   │   ├── toPrimitive.js.map
│   │   │   │   │   │   ├── toPropertyKey.js
│   │   │   │   │   │   ├── toPropertyKey.js.map
│   │   │   │   │   │   ├── toSetter.js
│   │   │   │   │   │   ├── toSetter.js.map
│   │   │   │   │   │   ├── tsRewriteRelativeImportExtensions.js
│   │   │   │   │   │   ├── tsRewriteRelativeImportExtensions.js.map
│   │   │   │   │   │   ├── typeof.js
│   │   │   │   │   │   ├── typeof.js.map
│   │   │   │   │   │   ├── unsupportedIterableToArray.js
│   │   │   │   │   │   ├── unsupportedIterableToArray.js.map
│   │   │   │   │   │   ├── using.js
│   │   │   │   │   │   ├── using.js.map
│   │   │   │   │   │   ├── usingCtx.js
│   │   │   │   │   │   ├── usingCtx.js.map
│   │   │   │   │   │   ├── wrapAsyncGenerator.js
│   │   │   │   │   │   ├── wrapAsyncGenerator.js.map
│   │   │   │   │   │   ├── wrapNativeSuper.js
│   │   │   │   │   │   ├── wrapNativeSuper.js.map
│   │   │   │   │   │   ├── wrapRegExp.js
│   │   │   │   │   │   ├── wrapRegExp.js.map
│   │   │   │   │   │   ├── writeOnlyError.js
│   │   │   │   │   │   └── writeOnlyError.js.map
│   │   │   │   │   ├── helpers-generated.js
│   │   │   │   │   ├── helpers-generated.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── parser
│   │   │   │   ├── CHANGELOG.md
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── bin
│   │   │   │   │   └── babel-parser.js
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   ├── package.json
│   │   │   │   └── typings
│   │   │   │       └── babel-parser.d.ts
│   │   │   ├── plugin-transform-react-jsx-self
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── plugin-transform-react-jsx-source
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── index.js
│   │   │   │   │   └── index.js.map
│   │   │   │   └── package.json
│   │   │   ├── template
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── formatters.js
│   │   │   │   │   ├── formatters.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── literal.js
│   │   │   │   │   ├── literal.js.map
│   │   │   │   │   ├── options.js
│   │   │   │   │   ├── options.js.map
│   │   │   │   │   ├── parse.js
│   │   │   │   │   ├── parse.js.map
│   │   │   │   │   ├── populate.js
│   │   │   │   │   ├── populate.js.map
│   │   │   │   │   ├── string.js
│   │   │   │   │   └── string.js.map
│   │   │   │   └── package.json
│   │   │   ├── traverse
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── lib
│   │   │   │   │   ├── cache.js
│   │   │   │   │   ├── cache.js.map
│   │   │   │   │   ├── context.js
│   │   │   │   │   ├── context.js.map
│   │   │   │   │   ├── hub.js
│   │   │   │   │   ├── hub.js.map
│   │   │   │   │   ├── index.js
│   │   │   │   │   ├── index.js.map
│   │   │   │   │   ├── path
│   │   │   │   │   │   ├── ancestry.js
│   │   │   │   │   │   ├── ancestry.js.map
│   │   │   │   │   │   ├── comments.js
│   │   │   │   │   │   ├── comments.js.map
│   │   │   │   │   │   ├── context.js
│   │   │   │   │   │   ├── context.js.map
│   │   │   │   │   │   ├── conversion.js
│   │   │   │   │   │   ├── conversion.js.map
│   │   │   │   │   │   ├── evaluation.js
│   │   │   │   │   │   ├── evaluation.js.map
│   │   │   │   │   │   ├── family.js
│   │   │   │   │   │   ├── family.js.map
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── inference
│   │   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   │   ├── inferer-reference.js
│   │   │   │   │   │   │   ├── inferer-reference.js.map
│   │   │   │   │   │   │   ├── inferers.js
│   │   │   │   │   │   │   ├── inferers.js.map
│   │   │   │   │   │   │   ├── util.js
│   │   │   │   │   │   │   └── util.js.map
│   │   │   │   │   │   ├── introspection.js
│   │   │   │   │   │   ├── introspection.js.map
│   │   │   │   │   │   ├── lib
│   │   │   │   │   │   │   ├── hoister.js
│   │   │   │   │   │   │   ├── hoister.js.map
│   │   │   │   │   │   │   ├── removal-hooks.js
│   │   │   │   │   │   │   ├── removal-hooks.js.map
│   │   │   │   │   │   │   ├── virtual-types-validator.js
│   │   │   │   │   │   │   ├── virtual-types-validator.js.map
│   │   │   │   │   │   │   ├── virtual-types.js
│   │   │   │   │   │   │   └── virtual-types.js.map
│   │   │   │   │   │   ├── modification.js
│   │   │   │   │   │   ├── modification.js.map
│   │   │   │   │   │   ├── removal.js
│   │   │   │   │   │   ├── removal.js.map
│   │   │   │   │   │   ├── replacement.js
│   │   │   │   │   │   └── replacement.js.map
│   │   │   │   │   ├── scope
│   │   │   │   │   │   ├── binding.js
│   │   │   │   │   │   ├── binding.js.map
│   │   │   │   │   │   ├── index.js
│   │   │   │   │   │   ├── index.js.map
│   │   │   │   │   │   ├── lib
│   │   │   │   │   │   │   ├── renamer.js
│   │   │   │   │   │   │   └── renamer.js.map
│   │   │   │   │   │   ├── traverseForScope.js
│   │   │   │   │   │   └── traverseForScope.js.map
│   │   │   │   │   ├── traverse-node.js
│   │   │   │   │   ├── traverse-node.js.map
│   │   │   │   │   ├── types.js
│   │   │   │   │   ├── types.js.map
│   │   │   │   │   ├── visitors.js
│   │   │   │   │   └── visitors.js.map
│   │   │   │   ├── package.json
│   │   │   │   └── tsconfig.overrides.json
│   │   │   └── types
│   │   │       ├── LICENSE
│   │   │       ├── README.md
│   │   │       ├── lib
│   │   │       │   ├── asserts
│   │   │       │   │   ├── assertNode.js
│   │   │       │   │   ├── assertNode.js.map
│   │   │       │   │   └── generated
│   │   │       │   │       ├── index.js
│   │   │       │   │       └── index.js.map
│   │   │       │   ├── ast-types
│   │   │       │   │   └── generated
│   │   │       │   │       ├── index.js
│   │   │       │   │       └── index.js.map
│   │   │       │   ├── clone
│   │   │       │   │   ├── clone.js
│   │   │       │   │   ├── clone.js.map
│   │   │       │   │   ├── cloneDeep.js
│   │   │       │   │   ├── cloneDeep.js.map
│   │   │       │   │   ├── cloneDeepWithoutLoc.js
│   │   │       │   │   ├── cloneDeepWithoutLoc.js.map
│   │   │       │   │   ├── cloneNode.js
│   │   │       │   │   ├── cloneNode.js.map
│   │   │       │   │   ├── cloneWithoutLoc.js
│   │   │       │   │   └── cloneWithoutLoc.js.map
│   │   │       │   ├── comments
│   │   │       │   │   ├── addComment.js
│   │   │       │   │   ├── addComment.js.map
│   │   │       │   │   ├── addComments.js
│   │   │       │   │   ├── addComments.js.map
│   │   │       │   │   ├── inheritInnerComments.js
│   │   │       │   │   ├── inheritInnerComments.js.map
│   │   │       │   │   ├── inheritLeadingComments.js
│   │   │       │   │   ├── inheritLeadingComments.js.map
│   │   │       │   │   ├── inheritTrailingComments.js
│   │   │       │   │   ├── inheritTrailingComments.js.map
│   │   │       │   │   ├── inheritsComments.js
│   │   │       │   │   ├── inheritsComments.js.map
│   │   │       │   │   ├── removeComments.js
│   │   │       │   │   └── removeComments.js.map
│   │   │       │   ├── constants
│   │   │       │   │   ├── generated
│   │   │       │   │   │   ├── index.js
│   │   │       │   │   │   └── index.js.map
│   │   │       │   │   ├── index.js
│   │   │       │   │   └── index.js.map
│   │   │       │   ├── converters
│   │   │       │   │   ├── ensureBlock.js
│   │   │       │   │   ├── ensureBlock.js.map
│   │   │       │   │   ├── gatherSequenceExpressions.js
│   │   │       │   │   ├── gatherSequenceExpressions.js.map
│   │   │       │   │   ├── toBindingIdentifierName.js
│   │   │       │   │   ├── toBindingIdentifierName.js.map
│   │   │       │   │   ├── toBlock.js
│   │   │       │   │   ├── toBlock.js.map
│   │   │       │   │   ├── toComputedKey.js
│   │   │       │   │   ├── toComputedKey.js.map
│   │   │       │   │   ├── toExpression.js
│   │   │       │   │   ├── toExpression.js.map
│   │   │       │   │   ├── toIdentifier.js
│   │   │       │   │   ├── toIdentifier.js.map
│   │   │       │   │   ├── toKeyAlias.js
│   │   │       │   │   ├── toKeyAlias.js.map
│   │   │       │   │   ├── toSequenceExpression.js
│   │   │       │   │   ├── toSequenceExpression.js.map
│   │   │       │   │   ├── toStatement.js
│   │   │       │   │   ├── toStatement.js.map
│   │   │       │   │   ├── valueToNode.js
│   │   │       │   │   └── valueToNode.js.map
│   │   │       │   ├── definitions
│   │   │       │   │   ├── core.js
│   │   │       │   │   ├── core.js.map
│   │   │       │   │   ├── deprecated-aliases.js
│   │   │       │   │   ├── deprecated-aliases.js.map
│   │   │       │   │   ├── experimental.js
│   │   │       │   │   ├── experimental.js.map
│   │   │       │   │   ├── flow.js
│   │   │       │   │   ├── flow.js.map
│   │   │       │   │   ├── index.js
│   │   │       │   │   ├── index.js.map
│   │   │       │   │   ├── jsx.js
│   │   │       │   │   ├── jsx.js.map
│   │   │       │   │   ├── misc.js
│   │   │       │   │   ├── misc.js.map
│   │   │       │   │   ├── placeholders.js
│   │   │       │   │   ├── placeholders.js.map
│   │   │       │   │   ├── typescript.js
│   │   │       │   │   ├── typescript.js.map
│   │   │       │   │   ├── utils.js
│   │   │       │   │   └── utils.js.map
│   │   │       │   ├── index-legacy.d.ts
│   │   │       │   ├── index.d.ts
│   │   │       │   ├── index.js
│   │   │       │   ├── index.js.flow
│   │   │       │   ├── index.js.map
│   │   │       │   ├── modifications
│   │   │       │   │   ├── appendToMemberExpression.js
│   │   │       │   │   ├── appendToMemberExpression.js.map
│   │   │       │   │   ├── flow
│   │   │       │   │   │   ├── removeTypeDuplicates.js
│   │   │       │   │   │   └── removeTypeDuplicates.js.map
│   │   │       │   │   ├── inherits.js
│   │   │       │   │   ├── inherits.js.map
│   │   │       │   │   ├── prependToMemberExpression.js
│   │   │       │   │   ├── prependToMemberExpression.js.map
│   │   │       │   │   ├── removeProperties.js
│   │   │       │   │   ├── removeProperties.js.map
│   │   │       │   │   ├── removePropertiesDeep.js
│   │   │       │   │   ├── removePropertiesDeep.js.map
│   │   │       │   │   └── typescript
│   │   │       │   │       ├── removeTypeDuplicates.js
│   │   │       │   │       └── removeTypeDuplicates.js.map
│   │   │       │   ├── retrievers
│   │   │       │   │   ├── getAssignmentIdentifiers.js
│   │   │       │   │   ├── getAssignmentIdentifiers.js.map
│   │   │       │   │   ├── getBindingIdentifiers.js
│   │   │       │   │   ├── getBindingIdentifiers.js.map
│   │   │       │   │   ├── getFunctionName.js
│   │   │       │   │   ├── getFunctionName.js.map
│   │   │       │   │   ├── getOuterBindingIdentifiers.js
│   │   │       │   │   └── getOuterBindingIdentifiers.js.map
│   │   │       │   ├── traverse
│   │   │       │   │   ├── traverse.js
│   │   │       │   │   ├── traverse.js.map
│   │   │       │   │   ├── traverseFast.js
│   │   │       │   │   └── traverseFast.js.map
│   │   │       │   ├── utils
│   │   │       │   │   ├── deprecationWarning.js
│   │   │       │   │   ├── deprecationWarning.js.map
│   │   │       │   │   ├── inherit.js
│   │   │       │   │   ├── inherit.js.map
│   │   │       │   │   ├── react
│   │   │       │   │   │   ├── cleanJSXElementLiteralChild.js
│   │   │       │   │   │   └── cleanJSXElementLiteralChild.js.map
│   │   │       │   │   ├── shallowEqual.js
│   │   │       │   │   └── shallowEqual.js.map
│   │   │       │   └── validators
│   │   │       │       ├── generated
│   │   │       │       │   ├── index.js
│   │   │       │       │   └── index.js.map
│   │   │       │       ├── is.js
│   │   │       │       ├── is.js.map
│   │   │       │       ├── isBinding.js
│   │   │       │       ├── isBinding.js.map
│   │   │       │       ├── isBlockScoped.js
│   │   │       │       ├── isBlockScoped.js.map
│   │   │       │       ├── isImmutable.js
│   │   │       │       ├── isImmutable.js.map
│   │   │       │       ├── isLet.js
│   │   │       │       ├── isLet.js.map
│   │   │       │       ├── isNode.js
│   │   │       │       ├── isNode.js.map
│   │   │       │       ├── isNodesEquivalent.js
│   │   │       │       ├── isNodesEquivalent.js.map
│   │   │       │       ├── isPlaceholderType.js
│   │   │       │       ├── isPlaceholderType.js.map
│   │   │       │       ├── isReferenced.js
│   │   │       │       ├── isReferenced.js.map
│   │   │       │       ├── isScope.js
│   │   │       │       ├── isScope.js.map
│   │   │       │       ├── isSpecifierDefault.js
│   │   │       │       ├── isSpecifierDefault.js.map
│   │   │       │       ├── isType.js
│   │   │       │       ├── isType.js.map
│   │   │       │       ├── isValidES3Identifier.js
│   │   │       │       ├── isValidES3Identifier.js.map
│   │   │       │       ├── isValidIdentifier.js
│   │   │       │       ├── isValidIdentifier.js.map
│   │   │       │       ├── isVar.js
│   │   │       │       ├── isVar.js.map
│   │   │       │       ├── matchesPattern.js
│   │   │       │       ├── matchesPattern.js.map
│   │   │       │       ├── react
│   │   │       │       │   ├── isCompatTag.js
│   │   │       │       │   ├── isCompatTag.js.map
│   │   │       │       │   ├── isReactComponent.js
│   │   │       │       │   └── isReactComponent.js.map
│   │   │       │       ├── validate.js
│   │   │       │       └── validate.js.map
│   │   │       └── package.json
│   │   ├── @jridgewell
│   │   │   ├── gen-mapping
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── package.json
│   │   │   │   ├── src
│   │   │   │   │   ├── gen-mapping.ts
│   │   │   │   │   ├── set-array.ts
│   │   │   │   │   ├── sourcemap-segment.ts
│   │   │   │   │   └── types.ts
│   │   │   │   └── types
│   │   │   │       ├── gen-mapping.d.cts
│   │   │   │       ├── gen-mapping.d.cts.map
│   │   │   │       ├── gen-mapping.d.mts
│   │   │   │       ├── gen-mapping.d.mts.map
│   │   │   │       ├── set-array.d.cts
│   │   │   │       ├── set-array.d.cts.map
│   │   │   │       ├── set-array.d.mts
│   │   │   │       ├── set-array.d.mts.map
│   │   │   │       ├── sourcemap-segment.d.cts
│   │   │   │       ├── sourcemap-segment.d.cts.map
│   │   │   │       ├── sourcemap-segment.d.mts
│   │   │   │       ├── sourcemap-segment.d.mts.map
│   │   │   │       ├── types.d.cts
│   │   │   │       ├── types.d.cts.map
│   │   │   │       ├── types.d.mts
│   │   │   │       └── types.d.mts.map
│   │   │   ├── remapping
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── package.json
│   │   │   │   ├── src
│   │   │   │   │   ├── remapping.ts
│   │   │   │   │   ├── source-map-tree.ts
│   │   │   │   │   ├── source-map.ts
│   │   │   │   │   └── types.ts
│   │   │   │   └── types
│   │   │   │       ├── remapping.d.cts
│   │   │   │       ├── remapping.d.cts.map
│   │   │   │       ├── remapping.d.mts
│   │   │   │       ├── remapping.d.mts.map
│   │   │   │       ├── source-map-tree.d.cts
│   │   │   │       ├── source-map-tree.d.cts.map
│   │   │   │       ├── source-map-tree.d.mts
│   │   │   │       ├── source-map-tree.d.mts.map
│   │   │   │       ├── source-map.d.cts
│   │   │   │       ├── source-map.d.cts.map
│   │   │   │       ├── source-map.d.mts
│   │   │   │       ├── source-map.d.mts.map
│   │   │   │       ├── types.d.cts
│   │   │   │       ├── types.d.cts.map
│   │   │   │       ├── types.d.mts
│   │   │   │       └── types.d.mts.map
│   │   │   ├── resolve-uri
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   └── package.json
│   │   │   ├── sourcemap-codec
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── package.json
│   │   │   │   ├── src
│   │   │   │   │   ├── scopes.ts
│   │   │   │   │   ├── sourcemap-codec.ts
│   │   │   │   │   ├── strings.ts
│   │   │   │   │   └── vlq.ts
│   │   │   │   └── types
│   │   │   │       ├── scopes.d.cts
│   │   │   │       ├── scopes.d.cts.map
│   │   │   │       ├── scopes.d.mts
│   │   │   │       ├── scopes.d.mts.map
│   │   │   │       ├── sourcemap-codec.d.cts
│   │   │   │       ├── sourcemap-codec.d.cts.map
│   │   │   │       ├── sourcemap-codec.d.mts
│   │   │   │       ├── sourcemap-codec.d.mts.map
│   │   │   │       ├── strings.d.cts
│   │   │   │       ├── strings.d.cts.map
│   │   │   │       ├── strings.d.mts
│   │   │   │       ├── strings.d.mts.map
│   │   │   │       ├── vlq.d.cts
│   │   │   │       ├── vlq.d.cts.map
│   │   │   │       ├── vlq.d.mts
│   │   │   │       └── vlq.d.mts.map
│   │   │   └── trace-mapping
│   │   │       ├── LICENSE
│   │   │       ├── README.md
│   │   │       ├── package.json
│   │   │       ├── src
│   │   │       │   ├── binary-search.ts
│   │   │       │   ├── by-source.ts
│   │   │       │   ├── flatten-map.ts
│   │   │       │   ├── resolve.ts
│   │   │       │   ├── sort.ts
│   │   │       │   ├── sourcemap-segment.ts
│   │   │       │   ├── strip-filename.ts
│   │   │       │   ├── trace-mapping.ts
│   │   │       │   └── types.ts
│   │   │       └── types
│   │   │           ├── binary-search.d.cts
│   │   │           ├── binary-search.d.cts.map
│   │   │           ├── binary-search.d.mts
│   │   │           ├── binary-search.d.mts.map
│   │   │           ├── by-source.d.cts
│   │   │           ├── by-source.d.cts.map
│   │   │           ├── by-source.d.mts
│   │   │           ├── by-source.d.mts.map
│   │   │           ├── flatten-map.d.cts
│   │   │           ├── flatten-map.d.cts.map
│   │   │           ├── flatten-map.d.mts
│   │   │           ├── flatten-map.d.mts.map
│   │   │           ├── resolve.d.cts
│   │   │           ├── resolve.d.cts.map
│   │   │           ├── resolve.d.mts
│   │   │           ├── resolve.d.mts.map
│   │   │           ├── sort.d.cts
│   │   │           ├── sort.d.cts.map
│   │   │           ├── sort.d.mts
│   │   │           ├── sort.d.mts.map
│   │   │           ├── sourcemap-segment.d.cts
│   │   │           ├── sourcemap-segment.d.cts.map
│   │   │           ├── sourcemap-segment.d.mts
│   │   │           ├── sourcemap-segment.d.mts.map
│   │   │           ├── strip-filename.d.cts
│   │   │           ├── strip-filename.d.cts.map
│   │   │           ├── strip-filename.d.mts
│   │   │           ├── strip-filename.d.mts.map
│   │   │           ├── trace-mapping.d.cts
│   │   │           ├── trace-mapping.d.cts.map
│   │   │           ├── trace-mapping.d.mts
│   │   │           ├── trace-mapping.d.mts.map
│   │   │           ├── types.d.cts
│   │   │           ├── types.d.cts.map
│   │   │           ├── types.d.mts
│   │   │           └── types.d.mts.map
│   │   ├── @rolldown
│   │   │   └── pluginutils
│   │   │       ├── LICENSE
│   │   │       └── package.json
│   │   ├── @rollup
│   │   │   ├── rollup-win32-x64-gnu
│   │   │   │   ├── README.md
│   │   │   │   ├── package.json
│   │   │   │   └── rollup.win32-x64-gnu.node
│   │   │   └── rollup-win32-x64-msvc
│   │   │       ├── README.md
│   │   │       ├── package.json
│   │   │       └── rollup.win32-x64-msvc.node
│   │   ├── @types
│   │   │   ├── babel__core
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── index.d.ts
│   │   │   │   └── package.json
│   │   │   ├── babel__generator
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── index.d.ts
│   │   │   │   └── package.json
│   │   │   ├── babel__template
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── index.d.ts
│   │   │   │   └── package.json
│   │   │   ├── babel__traverse
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── index.d.ts
│   │   │   │   └── package.json
│   │   │   └── estree
│   │   │       ├── LICENSE
│   │   │       ├── README.md
│   │   │       ├── flow.d.ts
│   │   │       ├── index.d.ts
│   │   │       └── package.json
│   │   ├── @vitejs
│   │   │   └── plugin-react
│   │   │       ├── LICENSE
│   │   │       ├── README.md
│   │   │       └── package.json
│   │   ├── baseline-browser-mapping
│   │   │   ├── LICENSE.txt
│   │   │   ├── README.md
│   │   │   └── package.json
│   │   ├── browserslist
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── browser.js
│   │   │   ├── cli.js
│   │   │   ├── error.d.ts
│   │   │   ├── error.js
│   │   │   ├── index.d.ts
│   │   │   ├── index.js
│   │   │   ├── node.js
│   │   │   ├── package.json
│   │   │   └── parse.js
│   │   ├── caniuse-lite
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── data
│   │   │   │   ├── agents.js
│   │   │   │   ├── browserVersions.js
│   │   │   │   ├── browsers.js
│   │   │   │   ├── features
│   │   │   │   │   ├── aac.js
│   │   │   │   │   ├── abortcontroller.js
│   │   │   │   │   ├── ac3-ec3.js
│   │   │   │   │   ├── accelerometer.js
│   │   │   │   │   ├── addeventlistener.js
│   │   │   │   │   ├── alternate-stylesheet.js
│   │   │   │   │   ├── ambient-light.js
│   │   │   │   │   ├── apng.js
│   │   │   │   │   ├── array-find-index.js
│   │   │   │   │   ├── array-find.js
│   │   │   │   │   ├── array-flat.js
│   │   │   │   │   ├── array-includes.js
│   │   │   │   │   ├── arrow-functions.js
│   │   │   │   │   ├── asmjs.js
│   │   │   │   │   ├── async-clipboard.js
│   │   │   │   │   ├── async-functions.js
│   │   │   │   │   ├── atob-btoa.js
│   │   │   │   │   ├── audio-api.js
│   │   │   │   │   ├── audio.js
│   │   │   │   │   ├── audiotracks.js
│   │   │   │   │   ├── autofocus.js
│   │   │   │   │   ├── auxclick.js
│   │   │   │   │   ├── av1.js
│   │   │   │   │   ├── avif.js
│   │   │   │   │   ├── background-attachment.js
│   │   │   │   │   ├── background-clip-text.js
│   │   │   │   │   ├── background-img-opts.js
│   │   │   │   │   ├── background-position-x-y.js
│   │   │   │   │   ├── background-repeat-round-space.js
│   │   │   │   │   ├── background-sync.js
│   │   │   │   │   ├── battery-status.js
│   │   │   │   │   ├── beacon.js
│   │   │   │   │   ├── beforeafterprint.js
│   │   │   │   │   ├── bigint.js
│   │   │   │   │   ├── bloburls.js
│   │   │   │   │   ├── border-image.js
│   │   │   │   │   ├── border-radius.js
│   │   │   │   │   ├── broadcastchannel.js
│   │   │   │   │   ├── brotli.js
│   │   │   │   │   ├── calc.js
│   │   │   │   │   ├── canvas-blending.js
│   │   │   │   │   ├── canvas-text.js
│   │   │   │   │   ├── canvas.js
│   │   │   │   │   ├── ch-unit.js
│   │   │   │   │   ├── chacha20-poly1305.js
│   │   │   │   │   ├── channel-messaging.js
│   │   │   │   │   ├── childnode-remove.js
│   │   │   │   │   ├── classlist.js
│   │   │   │   │   ├── client-hints-dpr-width-viewport.js
│   │   │   │   │   ├── clipboard.js
│   │   │   │   │   ├── colr-v1.js
│   │   │   │   │   ├── colr.js
│   │   │   │   │   ├── comparedocumentposition.js
│   │   │   │   │   ├── console-basic.js
│   │   │   │   │   ├── console-time.js
│   │   │   │   │   ├── const.js
│   │   │   │   │   ├── constraint-validation.js
│   │   │   │   │   ├── contenteditable.js
│   │   │   │   │   ├── contentsecuritypolicy.js
│   │   │   │   │   ├── contentsecuritypolicy2.js
│   │   │   │   │   ├── cookie-store-api.js
│   │   │   │   │   ├── cors.js
│   │   │   │   │   ├── createimagebitmap.js
│   │   │   │   │   ├── credential-management.js
│   │   │   │   │   ├── cross-document-view-transitions.js
│   │   │   │   │   ├── cryptography.js
│   │   │   │   │   ├── css-all.js
│   │   │   │   │   ├── css-anchor-positioning.js
│   │   │   │   │   ├── css-animation.js
│   │   │   │   │   ├── css-any-link.js
│   │   │   │   │   ├── css-appearance.js
│   │   │   │   │   ├── css-at-counter-style.js
│   │   │   │   │   ├── css-autofill.js
│   │   │   │   │   ├── css-backdrop-filter.js
│   │   │   │   │   ├── css-background-offsets.js
│   │   │   │   │   ├── css-backgroundblendmode.js
│   │   │   │   │   ├── css-boxdecorationbreak.js
│   │   │   │   │   ├── css-boxshadow.js
│   │   │   │   │   ├── css-canvas.js
│   │   │   │   │   ├── css-caret-color.js
│   │   │   │   │   ├── css-cascade-layers.js
│   │   │   │   │   ├── css-cascade-scope.js
│   │   │   │   │   ├── css-case-insensitive.js
│   │   │   │   │   ├── css-clip-path.js
│   │   │   │   │   ├── css-color-adjust.js
│   │   │   │   │   ├── css-color-function.js
│   │   │   │   │   ├── css-conic-gradients.js
│   │   │   │   │   ├── css-container-queries-style.js
│   │   │   │   │   ├── css-container-queries.js
│   │   │   │   │   ├── css-container-query-units.js
│   │   │   │   │   ├── css-containment.js
│   │   │   │   │   ├── css-content-visibility.js
│   │   │   │   │   ├── css-counters.js
│   │   │   │   │   ├── css-crisp-edges.js
│   │   │   │   │   ├── css-cross-fade.js
│   │   │   │   │   ├── css-default-pseudo.js
│   │   │   │   │   ├── css-descendant-gtgt.js
│   │   │   │   │   ├── css-deviceadaptation.js
│   │   │   │   │   ├── css-dir-pseudo.js
│   │   │   │   │   ├── css-display-contents.js
│   │   │   │   │   ├── css-element-function.js
│   │   │   │   │   ├── css-env-function.js
│   │   │   │   │   ├── css-exclusions.js
│   │   │   │   │   ├── css-featurequeries.js
│   │   │   │   │   ├── css-file-selector-button.js
│   │   │   │   │   ├── css-filter-function.js
│   │   │   │   │   ├── css-filters.js
│   │   │   │   │   ├── css-first-letter.js
│   │   │   │   │   ├── css-first-line.js
│   │   │   │   │   ├── css-fixed.js
│   │   │   │   │   ├── css-focus-visible.js
│   │   │   │   │   ├── css-focus-within.js
│   │   │   │   │   ├── css-font-palette.js
│   │   │   │   │   ├── css-font-rendering-controls.js
│   │   │   │   │   ├── css-font-stretch.js
│   │   │   │   │   ├── css-gencontent.js
│   │   │   │   │   ├── css-gradients.js
│   │   │   │   │   ├── css-grid-animation.js
│   │   │   │   │   ├── css-grid-lanes.js
│   │   │   │   │   ├── css-grid.js
│   │   │   │   │   ├── css-hanging-punctuation.js
│   │   │   │   │   ├── css-has.js
│   │   │   │   │   ├── css-hyphens.js
│   │   │   │   │   ├── css-if.js
│   │   │   │   │   ├── css-image-orientation.js
│   │   │   │   │   ├── css-image-set.js
│   │   │   │   │   ├── css-in-out-of-range.js
│   │   │   │   │   ├── css-indeterminate-pseudo.js
│   │   │   │   │   ├── css-initial-letter.js
│   │   │   │   │   ├── css-initial-value.js
│   │   │   │   │   ├── css-lch-lab.js
│   │   │   │   │   ├── css-letter-spacing.js
│   │   │   │   │   ├── css-line-clamp.js
│   │   │   │   │   ├── css-logical-props.js
│   │   │   │   │   ├── css-marker-pseudo.js
│   │   │   │   │   ├── css-masks.js
│   │   │   │   │   ├── css-matches-pseudo.js
│   │   │   │   │   ├── css-math-functions.js
│   │   │   │   │   ├── css-media-interaction.js
│   │   │   │   │   ├── css-media-range-syntax.js
│   │   │   │   │   ├── css-media-resolution.js
│   │   │   │   │   ├── css-media-scripting.js
│   │   │   │   │   ├── css-mediaqueries.js
│   │   │   │   │   ├── css-mixblendmode.js
│   │   │   │   │   ├── css-module-scripts.js
│   │   │   │   │   ├── css-motion-paths.js
│   │   │   │   │   ├── css-namespaces.js
│   │   │   │   │   ├── css-nesting.js
│   │   │   │   │   ├── css-not-sel-list.js
│   │   │   │   │   ├── css-nth-child-of.js
│   │   │   │   │   ├── css-opacity.js
│   │   │   │   │   ├── css-optional-pseudo.js
│   │   │   │   │   ├── css-overflow-anchor.js
│   │   │   │   │   ├── css-overflow-overlay.js
│   │   │   │   │   ├── css-overflow.js
│   │   │   │   │   ├── css-overscroll-behavior.js
│   │   │   │   │   ├── css-page-break.js
│   │   │   │   │   ├── css-paged-media.js
│   │   │   │   │   ├── css-paint-api.js
│   │   │   │   │   ├── css-placeholder-shown.js
│   │   │   │   │   ├── css-placeholder.js
│   │   │   │   │   ├── css-print-color-adjust.js
│   │   │   │   │   ├── css-read-only-write.js
│   │   │   │   │   ├── css-rebeccapurple.js
│   │   │   │   │   ├── css-reflections.js
│   │   │   │   │   ├── css-regions.js
│   │   │   │   │   ├── css-relative-colors.js
│   │   │   │   │   ├── css-repeating-gradients.js
│   │   │   │   │   ├── css-resize.js
│   │   │   │   │   ├── css-revert-value.js
│   │   │   │   │   ├── css-rrggbbaa.js
│   │   │   │   │   ├── css-scroll-behavior.js
│   │   │   │   │   ├── css-scrollbar.js
│   │   │   │   │   ├── css-sel2.js
│   │   │   │   │   ├── css-sel3.js
│   │   │   │   │   ├── css-selection.js
│   │   │   │   │   ├── css-shapes.js
│   │   │   │   │   ├── css-snappoints.js
│   │   │   │   │   ├── css-sticky.js
│   │   │   │   │   ├── css-subgrid.js
│   │   │   │   │   ├── css-supports-api.js
│   │   │   │   │   ├── css-table.js
│   │   │   │   │   ├── css-text-align-last.js
│   │   │   │   │   ├── css-text-box-trim.js
│   │   │   │   │   ├── css-text-indent.js
│   │   │   │   │   ├── css-text-justify.js
│   │   │   │   │   ├── css-text-orientation.js
│   │   │   │   │   ├── css-text-spacing.js
│   │   │   │   │   ├── css-text-wrap-balance.js
│   │   │   │   │   ├── css-textshadow.js
│   │   │   │   │   ├── css-touch-action.js
│   │   │   │   │   ├── css-transitions.js
│   │   │   │   │   ├── css-unicode-bidi.js
│   │   │   │   │   ├── css-unset-value.js
│   │   │   │   │   ├── css-variables.js
│   │   │   │   │   ├── css-when-else.js
│   │   │   │   │   ├── css-widows-orphans.js
│   │   │   │   │   ├── css-width-stretch.js
│   │   │   │   │   ├── css-writing-mode.js
│   │   │   │   │   ├── css-zoom.js
│   │   │   │   │   ├── css3-attr.js
│   │   │   │   │   ├── css3-boxsizing.js
│   │   │   │   │   ├── css3-colors.js
│   │   │   │   │   ├── css3-cursors-grab.js
│   │   │   │   │   ├── css3-cursors-newer.js
│   │   │   │   │   ├── css3-cursors.js
│   │   │   │   │   ├── css3-tabsize.js
│   │   │   │   │   ├── currentcolor.js
│   │   │   │   │   ├── custom-elements.js
│   │   │   │   │   ├── custom-elementsv1.js
│   │   │   │   │   ├── customevent.js
│   │   │   │   │   ├── customizable-select.js
│   │   │   │   │   ├── datalist.js
│   │   │   │   │   ├── dataset.js
│   │   │   │   │   ├── datauri.js
│   │   │   │   │   ├── date-tolocaledatestring.js
│   │   │   │   │   ├── declarative-shadow-dom.js
│   │   │   │   │   ├── decorators.js
│   │   │   │   │   ├── details.js
│   │   │   │   │   ├── deviceorientation.js
│   │   │   │   │   ├── devicepixelratio.js
│   │   │   │   │   ├── dialog.js
│   │   │   │   │   ├── dispatchevent.js
│   │   │   │   │   ├── dnssec.js
│   │   │   │   │   ├── do-not-track.js
│   │   │   │   │   ├── document-currentscript.js
│   │   │   │   │   ├── document-evaluate-xpath.js
│   │   │   │   │   ├── document-execcommand.js
│   │   │   │   │   ├── document-policy.js
│   │   │   │   │   ├── document-scrollingelement.js
│   │   │   │   │   ├── documenthead.js
│   │   │   │   │   ├── dom-manip-convenience.js
│   │   │   │   │   ├── dom-range.js
│   │   │   │   │   ├── domcontentloaded.js
│   │   │   │   │   ├── dommatrix.js
│   │   │   │   │   ├── download.js
│   │   │   │   │   ├── dragndrop.js
│   │   │   │   │   ├── element-closest.js
│   │   │   │   │   ├── element-from-point.js
│   │   │   │   │   ├── element-scroll-methods.js
│   │   │   │   │   ├── eme.js
│   │   │   │   │   ├── eot.js
│   │   │   │   │   ├── es5.js
│   │   │   │   │   ├── es6-class.js
│   │   │   │   │   ├── es6-generators.js
│   │   │   │   │   ├── es6-module-dynamic-import.js
│   │   │   │   │   ├── es6-module.js
│   │   │   │   │   ├── es6-number.js
│   │   │   │   │   ├── es6-string-includes.js
│   │   │   │   │   ├── es6.js
│   │   │   │   │   ├── eventsource.js
│   │   │   │   │   ├── extended-system-fonts.js
│   │   │   │   │   ├── feature-policy.js
│   │   │   │   │   ├── fetch.js
│   │   │   │   │   ├── fieldset-disabled.js
│   │   │   │   │   ├── fileapi.js
│   │   │   │   │   ├── filereader.js
│   │   │   │   │   ├── filereadersync.js
│   │   │   │   │   ├── filesystem.js
│   │   │   │   │   ├── flac.js
│   │   │   │   │   ├── flexbox-gap.js
│   │   │   │   │   ├── flexbox.js
│   │   │   │   │   ├── flow-root.js
│   │   │   │   │   ├── focusin-focusout-events.js
│   │   │   │   │   ├── font-family-system-ui.js
│   │   │   │   │   ├── font-feature.js
│   │   │   │   │   ├── font-kerning.js
│   │   │   │   │   ├── font-loading.js
│   │   │   │   │   ├── font-size-adjust.js
│   │   │   │   │   ├── font-smooth.js
│   │   │   │   │   ├── font-unicode-range.js
│   │   │   │   │   ├── font-variant-alternates.js
│   │   │   │   │   ├── font-variant-numeric.js
│   │   │   │   │   ├── fontface.js
│   │   │   │   │   ├── form-attribute.js
│   │   │   │   │   ├── form-submit-attributes.js
│   │   │   │   │   ├── form-validation.js
│   │   │   │   │   ├── forms.js
│   │   │   │   │   ├── fullscreen.js
│   │   │   │   │   ├── gamepad.js
│   │   │   │   │   ├── geolocation.js
│   │   │   │   │   ├── getboundingclientrect.js
│   │   │   │   │   ├── getcomputedstyle.js
│   │   │   │   │   ├── getelementsbyclassname.js
│   │   │   │   │   ├── getrandomvalues.js
│   │   │   │   │   ├── gyroscope.js
│   │   │   │   │   ├── hardwareconcurrency.js
│   │   │   │   │   ├── hashchange.js
│   │   │   │   │   ├── heif.js
│   │   │   │   │   ├── hevc.js
│   │   │   │   │   ├── hidden.js
│   │   │   │   │   ├── high-resolution-time.js
│   │   │   │   │   ├── history.js
│   │   │   │   │   ├── html-media-capture.js
│   │   │   │   │   ├── html5semantic.js
│   │   │   │   │   ├── http-live-streaming.js
│   │   │   │   │   ├── http2.js
│   │   │   │   │   ├── http3.js
│   │   │   │   │   ├── iframe-sandbox.js
│   │   │   │   │   ├── iframe-seamless.js
│   │   │   │   │   ├── iframe-srcdoc.js
│   │   │   │   │   ├── imagecapture.js
│   │   │   │   │   ├── ime.js
│   │   │   │   │   ├── img-naturalwidth-naturalheight.js
│   │   │   │   │   ├── import-maps.js
│   │   │   │   │   ├── imports.js
│   │   │   │   │   ├── indeterminate-checkbox.js
│   │   │   │   │   ├── indexeddb.js
│   │   │   │   │   ├── indexeddb2.js
│   │   │   │   │   ├── inline-block.js
│   │   │   │   │   ├── innertext.js
│   │   │   │   │   ├── input-autocomplete-onoff.js
│   │   │   │   │   ├── input-color.js
│   │   │   │   │   ├── input-datetime.js
│   │   │   │   │   ├── input-email-tel-url.js
│   │   │   │   │   ├── input-event.js
│   │   │   │   │   ├── input-file-accept.js
│   │   │   │   │   ├── input-file-directory.js
│   │   │   │   │   ├── input-file-multiple.js
│   │   │   │   │   ├── input-inputmode.js
│   │   │   │   │   ├── input-minlength.js
│   │   │   │   │   ├── input-number.js
│   │   │   │   │   ├── input-pattern.js
│   │   │   │   │   ├── input-placeholder.js
│   │   │   │   │   ├── input-range.js
│   │   │   │   │   ├── input-search.js
│   │   │   │   │   ├── input-selection.js
│   │   │   │   │   ├── insert-adjacent.js
│   │   │   │   │   ├── insertadjacenthtml.js
│   │   │   │   │   ├── internationalization.js
│   │   │   │   │   ├── intersectionobserver-v2.js
│   │   │   │   │   ├── intersectionobserver.js
│   │   │   │   │   ├── intl-pluralrules.js
│   │   │   │   │   ├── intrinsic-width.js
│   │   │   │   │   ├── jpeg2000.js
│   │   │   │   │   ├── jpegxl.js
│   │   │   │   │   ├── jpegxr.js
│   │   │   │   │   ├── js-regexp-lookbehind.js
│   │   │   │   │   ├── json.js
│   │   │   │   │   ├── justify-content-space-evenly.js
│   │   │   │   │   ├── kerning-pairs-ligatures.js
│   │   │   │   │   ├── keyboardevent-charcode.js
│   │   │   │   │   ├── keyboardevent-code.js
│   │   │   │   │   ├── keyboardevent-getmodifierstate.js
│   │   │   │   │   ├── keyboardevent-key.js
│   │   │   │   │   ├── keyboardevent-location.js
│   │   │   │   │   ├── keyboardevent-which.js
│   │   │   │   │   ├── lazyload.js
│   │   │   │   │   ├── let.js
│   │   │   │   │   ├── link-icon-png.js
│   │   │   │   │   ├── link-icon-svg.js
│   │   │   │   │   ├── link-rel-dns-prefetch.js
│   │   │   │   │   ├── link-rel-modulepreload.js
│   │   │   │   │   ├── link-rel-preconnect.js
│   │   │   │   │   ├── link-rel-prefetch.js
│   │   │   │   │   ├── link-rel-preload.js
│   │   │   │   │   ├── link-rel-prerender.js
│   │   │   │   │   ├── loading-lazy-attr.js
│   │   │   │   │   ├── loading-lazy-media.js
│   │   │   │   │   ├── localecompare.js
│   │   │   │   │   ├── magnetometer.js
│   │   │   │   │   ├── matchesselector.js
│   │   │   │   │   ├── matchmedia.js
│   │   │   │   │   ├── mathml.js
│   │   │   │   │   ├── maxlength.js
│   │   │   │   │   ├── mdn-css-backdrop-pseudo-element.js
│   │   │   │   │   ├── mdn-css-unicode-bidi-isolate-override.js
│   │   │   │   │   ├── mdn-css-unicode-bidi-isolate.js
│   │   │   │   │   ├── mdn-css-unicode-bidi-plaintext.js
│   │   │   │   │   ├── mdn-text-decoration-color.js
│   │   │   │   │   ├── mdn-text-decoration-line.js
│   │   │   │   │   ├── mdn-text-decoration-shorthand.js
│   │   │   │   │   ├── mdn-text-decoration-style.js
│   │   │   │   │   ├── media-fragments.js
│   │   │   │   │   ├── mediacapture-fromelement.js
│   │   │   │   │   ├── mediarecorder.js
│   │   │   │   │   ├── mediasource.js
│   │   │   │   │   ├── menu.js
│   │   │   │   │   ├── meta-theme-color.js
│   │   │   │   │   ├── meter.js
│   │   │   │   │   ├── midi.js
│   │   │   │   │   ├── minmaxwh.js
│   │   │   │   │   ├── mp3.js
│   │   │   │   │   ├── mpeg-dash.js
│   │   │   │   │   ├── mpeg4.js
│   │   │   │   │   ├── multibackgrounds.js
│   │   │   │   │   ├── multicolumn.js
│   │   │   │   │   ├── mutation-events.js
│   │   │   │   │   ├── mutationobserver.js
│   │   │   │   │   ├── namevalue-storage.js
│   │   │   │   │   ├── native-filesystem-api.js
│   │   │   │   │   ├── nav-timing.js
│   │   │   │   │   ├── netinfo.js
│   │   │   │   │   ├── notifications.js
│   │   │   │   │   ├── object-entries.js
│   │   │   │   │   ├── object-fit.js
│   │   │   │   │   ├── object-observe.js
│   │   │   │   │   ├── object-values.js
│   │   │   │   │   ├── objectrtc.js
│   │   │   │   │   ├── offline-apps.js
│   │   │   │   │   ├── offscreencanvas.js
│   │   │   │   │   ├── ogg-vorbis.js
│   │   │   │   │   ├── ogv.js
│   │   │   │   │   ├── ol-reversed.js
│   │   │   │   │   ├── once-event-listener.js
│   │   │   │   │   ├── online-status.js
│   │   │   │   │   ├── opus.js
│   │   │   │   │   ├── orientation-sensor.js
│   │   │   │   │   ├── outline.js
│   │   │   │   │   ├── pad-start-end.js
│   │   │   │   │   ├── page-transition-events.js
│   │   │   │   │   ├── pagevisibility.js
│   │   │   │   │   ├── passive-event-listener.js
│   │   │   │   │   ├── passkeys.js
│   │   │   │   │   ├── passwordrules.js
│   │   │   │   │   ├── path2d.js
│   │   │   │   │   ├── payment-request.js
│   │   │   │   │   ├── pdf-viewer.js
│   │   │   │   │   ├── permissions-api.js
│   │   │   │   │   ├── permissions-policy.js
│   │   │   │   │   ├── picture-in-picture.js
│   │   │   │   │   ├── picture.js
│   │   │   │   │   ├── ping.js
│   │   │   │   │   ├── png-alpha.js
│   │   │   │   │   ├── pointer-events.js
│   │   │   │   │   ├── pointer.js
│   │   │   │   │   ├── pointerlock.js
│   │   │   │   │   ├── portals.js
│   │   │   │   │   ├── prefers-color-scheme.js
│   │   │   │   │   ├── prefers-reduced-motion.js
│   │   │   │   │   ├── progress.js
│   │   │   │   │   ├── promise-finally.js
│   │   │   │   │   ├── promises.js
│   │   │   │   │   ├── proximity.js
│   │   │   │   │   ├── proxy.js
│   │   │   │   │   ├── publickeypinning.js
│   │   │   │   │   ├── push-api.js
│   │   │   │   │   ├── queryselector.js
│   │   │   │   │   ├── readonly-attr.js
│   │   │   │   │   ├── referrer-policy.js
│   │   │   │   │   ├── registerprotocolhandler.js
│   │   │   │   │   ├── rel-noopener.js
│   │   │   │   │   ├── rel-noreferrer.js
│   │   │   │   │   ├── rellist.js
│   │   │   │   │   ├── rem.js
│   │   │   │   │   ├── requestanimationframe.js
│   │   │   │   │   ├── requestidlecallback.js
│   │   │   │   │   ├── resizeobserver.js
│   │   │   │   │   ├── resource-timing.js
│   │   │   │   │   ├── rest-parameters.js
│   │   │   │   │   ├── rtcpeerconnection.js
│   │   │   │   │   ├── ruby.js
│   │   │   │   │   ├── run-in.js
│   │   │   │   │   ├── same-site-cookie-attribute.js
│   │   │   │   │   ├── screen-orientation.js
│   │   │   │   │   ├── script-async.js
│   │   │   │   │   ├── script-defer.js
│   │   │   │   │   ├── scrollintoview.js
│   │   │   │   │   ├── scrollintoviewifneeded.js
│   │   │   │   │   ├── sdch.js
│   │   │   │   │   ├── selection-api.js
│   │   │   │   │   ├── server-timing.js
│   │   │   │   │   ├── serviceworkers.js
│   │   │   │   │   ├── setimmediate.js
│   │   │   │   │   ├── shadowdom.js
│   │   │   │   │   ├── shadowdomv1.js
│   │   │   │   │   ├── sharedarraybuffer.js
│   │   │   │   │   ├── sharedworkers.js
│   │   │   │   │   ├── sni.js
│   │   │   │   │   ├── spdy.js
│   │   │   │   │   ├── speech-recognition.js
│   │   │   │   │   ├── speech-synthesis.js
│   │   │   │   │   ├── spellcheck-attribute.js
│   │   │   │   │   ├── sql-storage.js
│   │   │   │   │   ├── srcset.js
│   │   │   │   │   ├── stream.js
│   │   │   │   │   ├── streams.js
│   │   │   │   │   ├── stricttransportsecurity.js
│   │   │   │   │   ├── style-scoped.js
│   │   │   │   │   ├── subresource-bundling.js
│   │   │   │   │   ├── subresource-integrity.js
│   │   │   │   │   ├── svg-css.js
│   │   │   │   │   ├── svg-filters.js
│   │   │   │   │   ├── svg-fonts.js
│   │   │   │   │   ├── svg-fragment.js
│   │   │   │   │   ├── svg-html.js
│   │   │   │   │   ├── svg-html5.js
│   │   │   │   │   ├── svg-img.js
│   │   │   │   │   ├── svg-smil.js
│   │   │   │   │   ├── svg.js
│   │   │   │   │   ├── sxg.js
│   │   │   │   │   ├── tabindex-attr.js
│   │   │   │   │   ├── template-literals.js
│   │   │   │   │   ├── template.js
│   │   │   │   │   ├── temporal.js
│   │   │   │   │   ├── testfeat.js
│   │   │   │   │   ├── text-decoration.js
│   │   │   │   │   ├── text-emphasis.js
│   │   │   │   │   ├── text-overflow.js
│   │   │   │   │   ├── text-size-adjust.js
│   │   │   │   │   ├── text-stroke.js
│   │   │   │   │   ├── textcontent.js
│   │   │   │   │   ├── textencoder.js
│   │   │   │   │   ├── tls1-1.js
│   │   │   │   │   ├── tls1-2.js
│   │   │   │   │   ├── tls1-3.js
│   │   │   │   │   ├── touch.js
│   │   │   │   │   ├── transforms2d.js
│   │   │   │   │   ├── transforms3d.js
│   │   │   │   │   ├── trusted-types.js
│   │   │   │   │   ├── ttf.js
│   │   │   │   │   ├── typedarrays.js
│   │   │   │   │   ├── u2f.js
│   │   │   │   │   ├── unhandledrejection.js
│   │   │   │   │   ├── upgradeinsecurerequests.js
│   │   │   │   │   ├── url-scroll-to-text-fragment.js
│   │   │   │   │   ├── url.js
│   │   │   │   │   ├── urlsearchparams.js
│   │   │   │   │   ├── use-strict.js
│   │   │   │   │   ├── user-select-none.js
│   │   │   │   │   ├── user-timing.js
│   │   │   │   │   ├── variable-fonts.js
│   │   │   │   │   ├── vector-effect.js
│   │   │   │   │   ├── vibration.js
│   │   │   │   │   ├── video.js
│   │   │   │   │   ├── videotracks.js
│   │   │   │   │   ├── view-transitions.js
│   │   │   │   │   ├── viewport-unit-variants.js
│   │   │   │   │   ├── viewport-units.js
│   │   │   │   │   ├── wai-aria.js
│   │   │   │   │   ├── wake-lock.js
│   │   │   │   │   ├── wasm-bigint.js
│   │   │   │   │   ├── wasm-bulk-memory.js
│   │   │   │   │   ├── wasm-extended-const.js
│   │   │   │   │   ├── wasm-gc.js
│   │   │   │   │   ├── wasm-multi-memory.js
│   │   │   │   │   ├── wasm-multi-value.js
│   │   │   │   │   ├── wasm-mutable-globals.js
│   │   │   │   │   ├── wasm-nontrapping-fptoint.js
│   │   │   │   │   ├── wasm-reference-types.js
│   │   │   │   │   ├── wasm-relaxed-simd.js
│   │   │   │   │   ├── wasm-signext.js
│   │   │   │   │   ├── wasm-simd.js
│   │   │   │   │   ├── wasm-tail-calls.js
│   │   │   │   │   ├── wasm-threads.js
│   │   │   │   │   ├── wasm.js
│   │   │   │   │   ├── wav.js
│   │   │   │   │   ├── wbr-element.js
│   │   │   │   │   ├── web-animation.js
│   │   │   │   │   ├── web-app-manifest.js
│   │   │   │   │   ├── web-bluetooth.js
│   │   │   │   │   ├── web-serial.js
│   │   │   │   │   ├── web-share.js
│   │   │   │   │   ├── webauthn.js
│   │   │   │   │   ├── webcodecs.js
│   │   │   │   │   ├── webgl.js
│   │   │   │   │   ├── webgl2.js
│   │   │   │   │   ├── webgpu.js
│   │   │   │   │   ├── webhid.js
│   │   │   │   │   ├── webkit-user-drag.js
│   │   │   │   │   ├── webm.js
│   │   │   │   │   ├── webnfc.js
│   │   │   │   │   ├── webp.js
│   │   │   │   │   ├── websockets.js
│   │   │   │   │   ├── webtransport.js
│   │   │   │   │   ├── webusb.js
│   │   │   │   │   ├── webvr.js
│   │   │   │   │   ├── webvtt.js
│   │   │   │   │   ├── webworkers.js
│   │   │   │   │   ├── webxr.js
│   │   │   │   │   ├── will-change.js
│   │   │   │   │   ├── woff.js
│   │   │   │   │   ├── woff2.js
│   │   │   │   │   ├── word-break.js
│   │   │   │   │   ├── wordwrap.js
│   │   │   │   │   ├── x-doc-messaging.js
│   │   │   │   │   ├── x-frame-options.js
│   │   │   │   │   ├── xhr2.js
│   │   │   │   │   ├── xhtml.js
│   │   │   │   │   ├── xhtmlsmil.js
│   │   │   │   │   ├── xml-serializer.js
│   │   │   │   │   └── zstd.js
│   │   │   │   ├── features.js
│   │   │   │   └── regions
│   │   │   │       ├── AD.js
│   │   │   │       ├── AE.js
│   │   │   │       ├── AF.js
│   │   │   │       ├── AG.js
│   │   │   │       ├── AI.js
│   │   │   │       ├── AL.js
│   │   │   │       ├── AM.js
│   │   │   │       ├── AO.js
│   │   │   │       ├── AR.js
│   │   │   │       ├── AS.js
│   │   │   │       ├── AT.js
│   │   │   │       ├── AU.js
│   │   │   │       ├── AW.js
│   │   │   │       ├── AX.js
│   │   │   │       ├── AZ.js
│   │   │   │       ├── BA.js
│   │   │   │       ├── BB.js
│   │   │   │       ├── BD.js
│   │   │   │       ├── BE.js
│   │   │   │       ├── BF.js
│   │   │   │       ├── BG.js
│   │   │   │       ├── BH.js
│   │   │   │       ├── BI.js
│   │   │   │       ├── BJ.js
│   │   │   │       ├── BM.js
│   │   │   │       ├── BN.js
│   │   │   │       ├── BO.js
│   │   │   │       ├── BR.js
│   │   │   │       ├── BS.js
│   │   │   │       ├── BT.js
│   │   │   │       ├── BW.js
│   │   │   │       ├── BY.js
│   │   │   │       ├── BZ.js
│   │   │   │       ├── CA.js
│   │   │   │       ├── CD.js
│   │   │   │       ├── CF.js
│   │   │   │       ├── CG.js
│   │   │   │       ├── CH.js
│   │   │   │       ├── CI.js
│   │   │   │       ├── CK.js
│   │   │   │       ├── CL.js
│   │   │   │       ├── CM.js
│   │   │   │       ├── CN.js
│   │   │   │       ├── CO.js
│   │   │   │       ├── CR.js
│   │   │   │       ├── CU.js
│   │   │   │       ├── CV.js
│   │   │   │       ├── CX.js
│   │   │   │       ├── CY.js
│   │   │   │       ├── CZ.js
│   │   │   │       ├── DE.js
│   │   │   │       ├── DJ.js
│   │   │   │       ├── DK.js
│   │   │   │       ├── DM.js
│   │   │   │       ├── DO.js
│   │   │   │       ├── DZ.js
│   │   │   │       ├── EC.js
│   │   │   │       ├── EE.js
│   │   │   │       ├── EG.js
│   │   │   │       ├── ER.js
│   │   │   │       ├── ES.js
│   │   │   │       ├── ET.js
│   │   │   │       ├── FI.js
│   │   │   │       ├── FJ.js
│   │   │   │       ├── FK.js
│   │   │   │       ├── FM.js
│   │   │   │       ├── FO.js
│   │   │   │       ├── FR.js
│   │   │   │       ├── GA.js
│   │   │   │       ├── GB.js
│   │   │   │       ├── GD.js
│   │   │   │       ├── GE.js
│   │   │   │       ├── GF.js
│   │   │   │       ├── GG.js
│   │   │   │       ├── GH.js
│   │   │   │       ├── GI.js
│   │   │   │       ├── GL.js
│   │   │   │       ├── GM.js
│   │   │   │       ├── GN.js
│   │   │   │       ├── GP.js
│   │   │   │       ├── GQ.js
│   │   │   │       ├── GR.js
│   │   │   │       ├── GT.js
│   │   │   │       ├── GU.js
│   │   │   │       ├── GW.js
│   │   │   │       ├── GY.js
│   │   │   │       ├── HK.js
│   │   │   │       ├── HN.js
│   │   │   │       ├── HR.js
│   │   │   │       ├── HT.js
│   │   │   │       ├── HU.js
│   │   │   │       ├── ID.js
│   │   │   │       ├── IE.js
│   │   │   │       ├── IL.js
│   │   │   │       ├── IM.js
│   │   │   │       ├── IN.js
│   │   │   │       ├── IQ.js
│   │   │   │       ├── IR.js
│   │   │   │       ├── IS.js
│   │   │   │       ├── IT.js
│   │   │   │       ├── JE.js
│   │   │   │       ├── JM.js
│   │   │   │       ├── JO.js
│   │   │   │       ├── JP.js
│   │   │   │       ├── KE.js
│   │   │   │       ├── KG.js
│   │   │   │       ├── KH.js
│   │   │   │       ├── KI.js
│   │   │   │       ├── KM.js
│   │   │   │       ├── KN.js
│   │   │   │       ├── KP.js
│   │   │   │       ├── KR.js
│   │   │   │       ├── KW.js
│   │   │   │       ├── KY.js
│   │   │   │       ├── KZ.js
│   │   │   │       ├── LA.js
│   │   │   │       ├── LB.js
│   │   │   │       ├── LC.js
│   │   │   │       ├── LI.js
│   │   │   │       ├── LK.js
│   │   │   │       ├── LR.js
│   │   │   │       ├── LS.js
│   │   │   │       ├── LT.js
│   │   │   │       ├── LU.js
│   │   │   │       ├── LV.js
│   │   │   │       ├── LY.js
│   │   │   │       ├── MA.js
│   │   │   │       ├── MC.js
│   │   │   │       ├── MD.js
│   │   │   │       ├── ME.js
│   │   │   │       ├── MG.js
│   │   │   │       ├── MH.js
│   │   │   │       ├── MK.js
│   │   │   │       ├── ML.js
│   │   │   │       ├── MM.js
│   │   │   │       ├── MN.js
│   │   │   │       ├── MO.js
│   │   │   │       ├── MP.js
│   │   │   │       ├── MQ.js
│   │   │   │       ├── MR.js
│   │   │   │       ├── MS.js
│   │   │   │       ├── MT.js
│   │   │   │       ├── MU.js
│   │   │   │       ├── MV.js
│   │   │   │       ├── MW.js
│   │   │   │       ├── MX.js
│   │   │   │       ├── MY.js
│   │   │   │       ├── MZ.js
│   │   │   │       ├── NA.js
│   │   │   │       ├── NC.js
│   │   │   │       ├── NE.js
│   │   │   │       ├── NF.js
│   │   │   │       ├── NG.js
│   │   │   │       ├── NI.js
│   │   │   │       ├── NL.js
│   │   │   │       ├── NO.js
│   │   │   │       ├── NP.js
│   │   │   │       ├── NR.js
│   │   │   │       ├── NU.js
│   │   │   │       ├── NZ.js
│   │   │   │       ├── OM.js
│   │   │   │       ├── PA.js
│   │   │   │       ├── PE.js
│   │   │   │       ├── PF.js
│   │   │   │       ├── PG.js
│   │   │   │       ├── PH.js
│   │   │   │       ├── PK.js
│   │   │   │       ├── PL.js
│   │   │   │       ├── PM.js
│   │   │   │       ├── PN.js
│   │   │   │       ├── PR.js
│   │   │   │       ├── PS.js
│   │   │   │       ├── PT.js
│   │   │   │       ├── PW.js
│   │   │   │       ├── PY.js
│   │   │   │       ├── QA.js
│   │   │   │       ├── RE.js
│   │   │   │       ├── RO.js
│   │   │   │       ├── RS.js
│   │   │   │       ├── RU.js
│   │   │   │       ├── RW.js
│   │   │   │       ├── SA.js
│   │   │   │       ├── SB.js
│   │   │   │       ├── SC.js
│   │   │   │       ├── SD.js
│   │   │   │       ├── SE.js
│   │   │   │       ├── SG.js
│   │   │   │       ├── SH.js
│   │   │   │       ├── SI.js
│   │   │   │       ├── SK.js
│   │   │   │       ├── SL.js
│   │   │   │       ├── SM.js
│   │   │   │       ├── SN.js
│   │   │   │       ├── SO.js
│   │   │   │       ├── SR.js
│   │   │   │       ├── ST.js
│   │   │   │       ├── SV.js
│   │   │   │       ├── SY.js
│   │   │   │       ├── SZ.js
│   │   │   │       ├── TC.js
│   │   │   │       ├── TD.js
│   │   │   │       ├── TG.js
│   │   │   │       ├── TH.js
│   │   │   │       ├── TJ.js
│   │   │   │       ├── TL.js
│   │   │   │       ├── TM.js
│   │   │   │       ├── TN.js
│   │   │   │       ├── TO.js
│   │   │   │       ├── TR.js
│   │   │   │       ├── TT.js
│   │   │   │       ├── TV.js
│   │   │   │       ├── TW.js
│   │   │   │       ├── TZ.js
│   │   │   │       ├── UA.js
│   │   │   │       ├── UG.js
│   │   │   │       ├── US.js
│   │   │   │       ├── UY.js
│   │   │   │       ├── UZ.js
│   │   │   │       ├── VA.js
│   │   │   │       ├── VC.js
│   │   │   │       ├── VE.js
│   │   │   │       ├── VG.js
│   │   │   │       ├── VI.js
│   │   │   │       ├── VN.js
│   │   │   │       ├── VU.js
│   │   │   │       ├── WF.js
│   │   │   │       ├── WS.js
│   │   │   │       ├── YE.js
│   │   │   │       ├── YT.js
│   │   │   │       ├── ZA.js
│   │   │   │       ├── ZM.js
│   │   │   │       ├── ZW.js
│   │   │   │       ├── alt-af.js
│   │   │   │       ├── alt-an.js
│   │   │   │       ├── alt-as.js
│   │   │   │       ├── alt-eu.js
│   │   │   │       ├── alt-na.js
│   │   │   │       ├── alt-oc.js
│   │   │   │       ├── alt-sa.js
│   │   │   │       └── alt-ww.js
│   │   │   └── package.json
│   │   ├── convert-source-map
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── index.js
│   │   │   └── package.json
│   │   ├── debug
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── package.json
│   │   │   └── src
│   │   │       ├── browser.js
│   │   │       ├── common.js
│   │   │       ├── index.js
│   │   │       └── node.js
│   │   ├── electron-to-chromium
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── chromium-versions.js
│   │   │   ├── chromium-versions.json
│   │   │   ├── full-chromium-versions.js
│   │   │   ├── full-chromium-versions.json
│   │   │   ├── full-versions.js
│   │   │   ├── full-versions.json
│   │   │   ├── index.js
│   │   │   ├── package.json
│   │   │   ├── versions.js
│   │   │   └── versions.json
│   │   ├── escalade
│   │   │   ├── index.d.mts
│   │   │   ├── index.d.ts
│   │   │   ├── license
│   │   │   ├── package.json
│   │   │   ├── readme.md
│   │   │   └── sync
│   │   │       ├── index.d.mts
│   │   │       ├── index.d.ts
│   │   │       ├── index.js
│   │   │       └── index.mjs
│   │   ├── gensync
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── index.js
│   │   │   ├── index.js.flow
│   │   │   ├── package.json
│   │   │   └── test
│   │   │       ├── .babelrc
│   │   │       └── index.test.js
│   │   ├── js-tokens
│   │   │   ├── CHANGELOG.md
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── index.js
│   │   │   └── package.json
│   │   ├── jsesc
│   │   │   ├── LICENSE-MIT.txt
│   │   │   ├── README.md
│   │   │   ├── bin
│   │   │   │   └── jsesc
│   │   │   ├── jsesc.js
│   │   │   ├── man
│   │   │   │   └── jsesc.1
│   │   │   └── package.json
│   │   ├── json5
│   │   │   ├── LICENSE.md
│   │   │   ├── README.md
│   │   │   ├── lib
│   │   │   │   ├── cli.js
│   │   │   │   ├── index.d.ts
│   │   │   │   ├── index.js
│   │   │   │   ├── parse.d.ts
│   │   │   │   ├── parse.js
│   │   │   │   ├── register.js
│   │   │   │   ├── require.js
│   │   │   │   ├── stringify.d.ts
│   │   │   │   ├── stringify.js
│   │   │   │   ├── unicode.d.ts
│   │   │   │   ├── unicode.js
│   │   │   │   ├── util.d.ts
│   │   │   │   └── util.js
│   │   │   └── package.json
│   │   ├── loose-envify
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── cli.js
│   │   │   ├── custom.js
│   │   │   ├── index.js
│   │   │   ├── loose-envify.js
│   │   │   ├── package.json
│   │   │   └── replace.js
│   │   ├── lru-cache
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── index.js
│   │   │   └── package.json
│   │   ├── ms
│   │   │   ├── index.js
│   │   │   ├── license.md
│   │   │   ├── package.json
│   │   │   └── readme.md
│   │   ├── nanoid
│   │   │   ├── .claude
│   │   │   │   └── settings.local.json
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── async
│   │   │   │   ├── index.browser.cjs
│   │   │   │   ├── index.browser.js
│   │   │   │   ├── index.cjs
│   │   │   │   ├── index.d.ts
│   │   │   │   ├── index.js
│   │   │   │   ├── index.native.js
│   │   │   │   └── package.json
│   │   │   ├── bin
│   │   │   │   └── nanoid.cjs
│   │   │   ├── index.browser.cjs
│   │   │   ├── index.browser.js
│   │   │   ├── index.cjs
│   │   │   ├── index.d.cts
│   │   │   ├── index.d.ts
│   │   │   ├── index.js
│   │   │   ├── nanoid.js
│   │   │   ├── non-secure
│   │   │   │   ├── index.cjs
│   │   │   │   ├── index.d.ts
│   │   │   │   ├── index.js
│   │   │   │   └── package.json
│   │   │   ├── package.json
│   │   │   └── url-alphabet
│   │   │       ├── index.cjs
│   │   │       ├── index.js
│   │   │       └── package.json
│   │   ├── node-releases
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── data
│   │   │   │   ├── processed
│   │   │   │   │   └── envs.json
│   │   │   │   └── release-schedule
│   │   │   │       └── release-schedule.json
│   │   │   └── package.json
│   │   ├── picocolors
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── package.json
│   │   │   ├── picocolors.browser.js
│   │   │   ├── picocolors.d.ts
│   │   │   ├── picocolors.js
│   │   │   └── types.d.ts
│   │   ├── postcss
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── lib
│   │   │   │   ├── at-rule.d.ts
│   │   │   │   ├── at-rule.js
│   │   │   │   ├── comment.d.ts
│   │   │   │   ├── comment.js
│   │   │   │   ├── container.d.ts
│   │   │   │   ├── container.js
│   │   │   │   ├── css-syntax-error.d.ts
│   │   │   │   ├── css-syntax-error.js
│   │   │   │   ├── declaration.d.ts
│   │   │   │   ├── declaration.js
│   │   │   │   ├── document.d.ts
│   │   │   │   ├── document.js
│   │   │   │   ├── fromJSON.d.ts
│   │   │   │   ├── fromJSON.js
│   │   │   │   ├── input.d.ts
│   │   │   │   ├── input.js
│   │   │   │   ├── lazy-result.d.ts
│   │   │   │   ├── lazy-result.js
│   │   │   │   ├── list.d.ts
│   │   │   │   ├── list.js
│   │   │   │   ├── map-generator.js
│   │   │   │   ├── no-work-result.d.ts
│   │   │   │   ├── no-work-result.js
│   │   │   │   ├── node.d.ts
│   │   │   │   ├── node.js
│   │   │   │   ├── parse.d.ts
│   │   │   │   ├── parse.js
│   │   │   │   ├── parser.js
│   │   │   │   ├── postcss.d.mts
│   │   │   │   ├── postcss.d.ts
│   │   │   │   ├── postcss.js
│   │   │   │   ├── postcss.mjs
│   │   │   │   ├── previous-map.d.ts
│   │   │   │   ├── previous-map.js
│   │   │   │   ├── processor.d.ts
│   │   │   │   ├── processor.js
│   │   │   │   ├── result.d.ts
│   │   │   │   ├── result.js
│   │   │   │   ├── root.d.ts
│   │   │   │   ├── root.js
│   │   │   │   ├── rule.d.ts
│   │   │   │   ├── rule.js
│   │   │   │   ├── stringifier.d.ts
│   │   │   │   ├── stringifier.js
│   │   │   │   ├── stringify.d.ts
│   │   │   │   ├── stringify.js
│   │   │   │   ├── symbols.js
│   │   │   │   ├── terminal-highlight.js
│   │   │   │   ├── tokenize.js
│   │   │   │   ├── warn-once.js
│   │   │   │   ├── warning.d.ts
│   │   │   │   └── warning.js
│   │   │   └── package.json
│   │   ├── react
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── cjs
│   │   │   │   ├── react-jsx-dev-runtime.development.js
│   │   │   │   ├── react-jsx-dev-runtime.production.min.js
│   │   │   │   ├── react-jsx-dev-runtime.profiling.min.js
│   │   │   │   ├── react-jsx-runtime.development.js
│   │   │   │   ├── react-jsx-runtime.production.min.js
│   │   │   │   ├── react-jsx-runtime.profiling.min.js
│   │   │   │   ├── react.development.js
│   │   │   │   ├── react.production.min.js
│   │   │   │   ├── react.shared-subset.development.js
│   │   │   │   └── react.shared-subset.production.min.js
│   │   │   ├── index.js
│   │   │   ├── jsx-dev-runtime.js
│   │   │   ├── jsx-runtime.js
│   │   │   ├── package.json
│   │   │   ├── react.shared-subset.js
│   │   │   └── umd
│   │   │       ├── react.development.js
│   │   │       ├── react.production.min.js
│   │   │       └── react.profiling.min.js
│   │   ├── react-dom
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── cjs
│   │   │   │   ├── react-dom-server-legacy.browser.development.js
│   │   │   │   ├── react-dom-server-legacy.browser.production.min.js
│   │   │   │   ├── react-dom-server-legacy.node.development.js
│   │   │   │   ├── react-dom-server-legacy.node.production.min.js
│   │   │   │   ├── react-dom-server.browser.development.js
│   │   │   │   ├── react-dom-server.browser.production.min.js
│   │   │   │   ├── react-dom-server.node.development.js
│   │   │   │   ├── react-dom-server.node.production.min.js
│   │   │   │   ├── react-dom-test-utils.development.js
│   │   │   │   ├── react-dom-test-utils.production.min.js
│   │   │   │   ├── react-dom.development.js
│   │   │   │   ├── react-dom.production.min.js
│   │   │   │   └── react-dom.profiling.min.js
│   │   │   ├── client.js
│   │   │   ├── index.js
│   │   │   ├── package.json
│   │   │   ├── profiling.js
│   │   │   ├── server.browser.js
│   │   │   ├── server.js
│   │   │   ├── server.node.js
│   │   │   ├── test-utils.js
│   │   │   └── umd
│   │   │       ├── react-dom-server-legacy.browser.development.js
│   │   │       ├── react-dom-server-legacy.browser.production.min.js
│   │   │       ├── react-dom-server.browser.development.js
│   │   │       ├── react-dom-server.browser.production.min.js
│   │   │       ├── react-dom-test-utils.development.js
│   │   │       ├── react-dom-test-utils.production.min.js
│   │   │       ├── react-dom.development.js
│   │   │       ├── react-dom.production.min.js
│   │   │       └── react-dom.profiling.min.js
│   │   ├── react-refresh
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── babel.js
│   │   │   ├── cjs
│   │   │   │   ├── react-refresh-babel.development.js
│   │   │   │   ├── react-refresh-babel.production.js
│   │   │   │   ├── react-refresh-runtime.development.js
│   │   │   │   └── react-refresh-runtime.production.js
│   │   │   ├── package.json
│   │   │   └── runtime.js
│   │   ├── rollup
│   │   │   ├── LICENSE.md
│   │   │   ├── README.md
│   │   │   └── package.json
│   │   ├── scheduler
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── cjs
│   │   │   │   ├── scheduler-unstable_mock.development.js
│   │   │   │   ├── scheduler-unstable_mock.production.min.js
│   │   │   │   ├── scheduler-unstable_post_task.development.js
│   │   │   │   ├── scheduler-unstable_post_task.production.min.js
│   │   │   │   ├── scheduler.development.js
│   │   │   │   └── scheduler.production.min.js
│   │   │   ├── index.js
│   │   │   ├── package.json
│   │   │   ├── umd
│   │   │   │   ├── scheduler-unstable_mock.development.js
│   │   │   │   ├── scheduler-unstable_mock.production.min.js
│   │   │   │   ├── scheduler.development.js
│   │   │   │   ├── scheduler.production.min.js
│   │   │   │   └── scheduler.profiling.min.js
│   │   │   ├── unstable_mock.js
│   │   │   └── unstable_post_task.js
│   │   ├── semver
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── bin
│   │   │   │   └── semver.js
│   │   │   ├── package.json
│   │   │   ├── range.bnf
│   │   │   └── semver.js
│   │   ├── source-map-js
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── lib
│   │   │   │   ├── array-set.js
│   │   │   │   ├── base64-vlq.js
│   │   │   │   ├── base64.js
│   │   │   │   ├── binary-search.js
│   │   │   │   ├── mapping-list.js
│   │   │   │   ├── quick-sort.js
│   │   │   │   ├── source-map-consumer.d.ts
│   │   │   │   ├── source-map-consumer.js
│   │   │   │   ├── source-map-generator.d.ts
│   │   │   │   ├── source-map-generator.js
│   │   │   │   ├── source-node.d.ts
│   │   │   │   ├── source-node.js
│   │   │   │   └── util.js
│   │   │   ├── package.json
│   │   │   ├── source-map.d.ts
│   │   │   └── source-map.js
│   │   ├── update-browserslist-db
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── check-npm-version.js
│   │   │   ├── cli.js
│   │   │   ├── index.d.ts
│   │   │   ├── index.js
│   │   │   ├── package.json
│   │   │   └── utils.js
│   │   ├── vite
│   │   │   ├── LICENSE.md
│   │   │   ├── README.md
│   │   │   ├── bin
│   │   │   │   ├── openChrome.applescript
│   │   │   │   └── vite.js
│   │   │   ├── client.d.ts
│   │   │   ├── index.cjs
│   │   │   ├── index.d.cts
│   │   │   ├── package.json
│   │   │   └── types
│   │   │       ├── customEvent.d.ts
│   │   │       ├── hmrPayload.d.ts
│   │   │       ├── hot.d.ts
│   │   │       ├── import-meta.d.ts
│   │   │       ├── importGlob.d.ts
│   │   │       ├── importMeta.d.ts
│   │   │       ├── metadata.d.ts
│   │   │       └── package.json
│   │   └── yallist
│   │       ├── LICENSE
│   │       ├── README.md
│   │       ├── iterator.js
│   │       ├── package.json
│   │       └── yallist.js
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── blocks
│   │   │   ├── Block1Structure.jsx
│   │   │   ├── Block2Epidemic.jsx
│   │   │   ├── Block3Polarization.jsx
│   │   │   ├── Block4Homophily.jsx
│   │   │   └── Block5Severity.jsx
│   │   ├── components
│   │   │   ├── LinePlot.jsx
│   │   │   ├── ModelSelect.jsx
│   │   │   ├── Slider.jsx
│   │   │   └── useDebouncedFetch.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   └── vite.config.js
├── notebooks
│   ├── RUNME.ipynb
│   ├── data
│   ├── data_homophily.csv
│   ├── helps.py
│   └── parameters.json
├── print_tree.ipynb
├── requirements.txt
├── setup.py
├── sir_model
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── contact.py
│   ├── loader.py
│   ├── models.py
│   ├── plotting.py
│   ├── simulate.py
│   └── sweep.py
└── tree.md
```