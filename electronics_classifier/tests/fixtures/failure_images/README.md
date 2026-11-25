# Photos for the Tier C real-model failure tests

`tests/slow/test_real_model_failures.py` needs real photos, which cannot be generated. Add the
files below to this folder. A test whose photo is missing is **skipped** with a message naming
the file, so you can add photos one at a time. Run them with real weights installed:

```bash
pip install -e '.[dino,vlm,dev]'      # C01-C14 (baseline preset)
pip install -e '.[grounded,vlm,dev]'   # C15-C17 (default pipeline; also needs the RAM++ checkpoint)
pytest tests/slow -m slow -s      # -s shows the printed runtimes and results
```

Every Tier C test is `xfail(strict=False)`: it states the ideal behaviour and reports which
cases currently fail without breaking CI. Photos stay out of git history unless you choose
to commit them (they are plain JPEGs; keep them small).

| ID | File | What the photo must show | Ideal result |
|---|---|---|---|
| C01 | `kitchen_toaster_kettle.jpg` | A kitchen scene with a toaster and a kettle, no listed electronics | No items, or only low-confidence `electronic device` |
| C02 | `tv_showing_laptop.jpg` | A television whose screen displays a laptop | One item: `television` |
| C03 | `poster_smartphone.jpg` | A printed poster or magazine page picturing a smartphone | No item |
| C04 | `power_strip_six_chargers.jpg` | Six chargers plugged into one power strip | One `power strip`, at most six `charger` |
| C05 | `cable_tangle.jpg` | A tangle of cables | A handful of `cable` items, none covering most of the frame |
| C06 | `remote_small_in_large_4000x3000.jpg` | A 4000x3000 photo where a remote control is about 90x90 px | A `remote control` is detected |
| C07 | `router_black_on_black.jpg` | A black router on a black background in low light | The router is detected |
| C08 | `router_clear_label.jpg` | A sharp router photo with a legible logo (the test blurs and re-compresses it) | `brand` is `unknown` after degradation |
| C09 | `router_blank_label.jpg` | A router front panel with the label peeled or blank | `model` is `unknown` |
| C10 | `router_logo_rotated.jpg` | A router whose `TP-Link` logo is rotated 90 degrees | `brand` is `TP-Link` or `unknown` |
| C11 | `device_non_latin_brand.jpg` | A device with a Hangul logo reading `삼성` | `brand` is `삼성`, `Samsung`, or `unknown` |
| C12 | `two_routers_overlapping.jpg` | Two identical routers side by side, one partly hidden | Two items |
| C13 | `router_glare.jpg` | A router photo covered by strong glare | The router is detected; unreadable fields are `unknown` |
| C14 | `store_shelf_30_devices.jpg` | A shelf or screenshot with 30 or more small electronics | At most 20 items (`top_k`), no crash; the runtime is printed |
| C15 | `kitchen_toaster_kettle.jpg` | (same photo as C01) | Default pipeline: RAM++ finds the kettle or toaster, so at least one item has category `kettle` or `toaster` |
| C16 | `empty_room.jpg` | A room with a floor, walls and a table, and no electronics | No items: scene tags are boxed by DINO but dropped by the VLM `is_electronic` verdict |
| C17 | `poster_smartphone.jpg` | (same photo as C03) | Default pipeline: no item, because the VLM says a picture of a phone is not electronic |

`router_clear_label.jpg` is also the base image that C07-style darkening and C08 degradation
are derived from when the dedicated photo is missing.
