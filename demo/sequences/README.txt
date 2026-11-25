For a new live clip, put its TUM-format folder here with rgb.txt, depth.txt,
groundtruth.txt, rgb\ and depth\ (16-bit PNG, 5000 units per metre).
Optional sequence.json: {"name": "My clip", "fx": 535.4, "fy": 539.2, "cx": 319.5, "cy": 239.5, "step": 2}.

Prepare the clip's depth with Fu et al. before the demo. Save the prepared
16-bit PNGs as <frame number>.png in demo\data\cache\<folder name>\paper__center_paper\.
The live picker shows clips with prepared frames. Ground truth is matched
by timestamp. Each live run computes ICP from the prepared depth.
