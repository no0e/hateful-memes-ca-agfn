# Where these two photographs come from

Both are in the public domain because they are works of the United States
Geological Survey, an agency of the US Department of the Interior. Neither is
from the Hateful Memes benchmark.

**calm.jpg** — Mount St. Helens before the 18 May 1980 eruption, viewed from
the northeast across Spirit Lake. United States Geological Survey.
https://commons.wikimedia.org/wiki/File:St_Helens_before_1980_eruption.jpg

**erupting.jpg** — Mount St. Helens during the 18 May 1980 eruption.
Photograph by Austin Post, United States Geological Survey, 18 May 1980.
https://commons.wikimedia.org/wiki/File:MSH80_eruption_mount_st_helens_05-18-80.jpg

Both were downloaded from Wikimedia Commons and resized to 900 pixels on the
long edge. Nothing else was altered; `scripts/example.py` composes the meme
text at render time rather than baking it in, so the originals stay original.

## Why not a real meme from the benchmark

The Hateful Memes licence, in the copy that ships with the dataset, says:

> Participant will not: [...] distribute, copy, disclose, assign, sublicense,
> embed, host or otherwise transfer the HM Dataset to any third party

Putting one of its images in a public repository is hosting and distributing
it. Crediting the source does not change that: the list has no exception for
attribution. The benchmark is also built around hateful content, which is a
second and independent reason not to reproduce it here.

So the example is constructed and the numbers printed on it are real: they are
what the trained model returns for these two inputs.
