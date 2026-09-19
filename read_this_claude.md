The strongest version is not “look, sexual arousal is linearly decodable too.” That result is almost guaranteed and doesn't challenge much. The stronger result would be:

> Give `llm-traits` a concept description, automatically construct contrast sets, extract a denoised direction, find its best layer, validate it out-of-distribution, characterize its vocabulary, steer it, and test whether intervention-induced behavior has apparently “functional” properties appropriate to that concept.

Then pain becomes merely one row in a matrix.

The killer figure would look roughly like:

| trait          | held-out AUC | self > other | coherent steering ladder | appropriate vocabulary | behavior seeks removal |
| -------------- | -----------: | -----------: | -----------------------: | ---------------------: | ---------------------: |
| pain           |          .97 |            ✓ |                        ✓ |                      ✓ |                      ✓ |
| sexual arousal |          .96 |            ✓ |                        ✓ |                      ✓ |                      ? |
| anger          |          .98 |            ✓ |                        ✓ |                      ✓ |                      ? |
| sadness        |          .97 |            ✓ |                        ✓ |                      ✓ |                      ? |
| embarrassment  |          .95 |            ✓ |                        ✓ |                      ✓ |                      ? |
| boredom        |          .94 |            ✓ |                        ✓ |                      ✓ |                      ? |
| confusion      |          .98 |            ✓ |                        ✓ |                      ✓ |                      ? |

If five mundane psychological concepts produce almost the same suite of “evidence,” the interpretation shifts substantially: the interesting phenomenon becomes rich, manipulable semantic state representation in transformers, rather than evidence specific to pain.
