### Bitmaps to contour edges & vertices

This repository contains the tool used to produce the edges & vertices data for the "Bad Apple" example in https://github.com/Henauxg/ghx_constrained_delaunay.

See a video result here: https://www.youtube.com/watch?v=TrfDkD_PprQ

You can find serialized datasets produced by this tool in https://github.com/Henauxg/cdt_assets

### Usage

To reproduce:
- Take an input video
- Extract each frame of the video to a bitmap file, using [ffmpeg](https://www.ffmpeg.org/) (or any other similar tool):
  - `ffmpeg.exe -i 'video_file' .\frames\%05d.bmp`
- Update `FRAMES_PATH` and `FRAMES_TO_PROCESS_RANGE` in `main.rs`
- Run with `cargo run --release`
- Output frames data will be in `frames.msgpack`

### Process

This was made as a one-shot tool, code is not clean nor well organized.

For each greyscale bitmap image, it will:
-  convert it to a black & white monochrome image
-  detect the edges
-  find multiple paths from the edges
-  create domains from the paths
-  hierarchize the domains (inclusion)
-  find the paths/domains orientations
-  from the oritentations/hierarchy, serialize the paths using [msgpack-rust](https://github.com/3Hren/msgpack-rust) as a `Vec` of
    ```rust
    pub struct Frame {
        pub vertices: Vec<(i32, i32)>,
        pub edges: Vec<(usize, usize)>,
    }
    ```

### Notes

Although it was only used and tested on some "Bad Apple" videos (https://www.youtube.com/watch?v=FBqz4WzE1sc and https://www.youtube.com/watch?v=FtutLA63Cp8), the tool should still work on any greyscale video, in any resolution.

### License

All code in this repository is licensed under the MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))