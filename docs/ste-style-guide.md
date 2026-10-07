# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of salescope. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **row** | One item-outlet pair of the BigMart file, with its sales | record, sample, observation |
| **item** | One product, named by `Item_Identifier` | product (alone), SKU, article |
| **outlet** | One store, named by `Outlet_Identifier` | store (alone), shop, branch |
| **target** | The column `Item_Outlet_Sales` | label, output, y |
| **feature** | One model input column that `BigMartFeatures` makes | variable, attribute, predictor |
| **schema** | The column contract in `schema.py` | spec, format |
| **pipeline** | The scikit-learn object: features, encoder and regressor | flow, chain |
| **model** | One entry of the model registry, as a full pipeline | algorithm, learner, estimator (in prose) |
| **baseline** | The `mean` or the `mrp_baseline` model | benchmark, naive model (for tabular data) |
| **CV scheme** | `random`, `new_outlet` or `new_item` | split type, strategy, validation mode |
| **fold** | One train part and one test part of a CV scheme | split (alone), partition |
| **group** | The outlet or the item that a grouped CV scheme keeps in one fold | cluster, block |
| **holdout** | The rows that one final score uses and no fit sees | test set (for this meaning), validation set |
| **OOF prediction** | A prediction for a row from a model that did not see the row | cross-validated prediction, in-sample prediction |
| **base model** | A model whose OOF predictions are inputs to the meta-model | level-0 model, first-stage model |
| **meta-model** | The non-negative linear regression that combines base models | blender, second-stage model |
| **stack** | The meta-model plus its base models | ensemble, hybrid |
| **series** | One dated monthly sales history, named by `series_id` | time series (in prose), sequence |
| **origin** | The first month that a backtest fold forecasts | cut-off, split date |
| **horizon** | The number of months after the origin that a forecaster predicts | window, lead time |
| **forecaster** | One model of the forecast track | predictor, time-series model |
| **synthetic data** | Data that `synthetic.py` or `make_series` makes | fake data, dummy data, mock data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check a frame against the schema and report each problem |
| **normalise** | Change the label variants of a column to one spelling |
| **impute** | Replace a missing value with a value that the training rows give |
| **fit** | Learn the parameters of a pipeline from the training rows |
| **predict** | Calculate the sales of rows with a fitted model |
| **evaluate** | Calculate the metrics of the OOF predictions under a CV scheme |
| **stack** | Fit the meta-model on OOF predictions and score it on the holdout |
| **tune** | Search the hyperparameters inside grouped CV |
| **backtest** | Fit forecasters before each origin and score the months after it |
| **clip** | Change each negative prediction to 0 |
