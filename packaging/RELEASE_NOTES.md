StopScrolling watches your webcam while you study. If you look at your phone for more than 1.5 seconds, it plays an escalating alarm and throws a new meme popup on screen every 0.1 seconds. All of them close the moment you put the phone down. Everything runs locally, and no video is saved or sent anywhere.

## Download

| Platform | File |
|---|---|
| Windows 10/11 (64-bit) | `StopScrolling-*-windows-x64.zip` |
| macOS (Apple Silicon) | `StopScrolling-*-macos-arm64.tar.gz` |
| Linux x64 (X11 desktop) | `StopScrolling-*-linux-x64.tar.gz` |

Extract the archive, then run the app:

- **Windows:** double-click `StopScrolling.exe`. SmartScreen may warn about an unrecognised app. Choose *More info* → *Run anyway*.
- **macOS:** the app isn't notarised. In Terminal, inside the extracted folder:
  ```
  xattr -dr com.apple.quarantine .
  ./StopScrolling
  ```
  Allow camera access for Terminal when asked.
- **Linux:** `./StopScrolling` from an X11 session.

A camera window opens. Press `q` to quit, or `s` to snooze alerts for 5 minutes. Run with `--help` to see all options, for example `--no-preview`, `--no-sound`, `--max-popups 15` and `--enter-s 3`.

To add your own memes and sounds, put them in the `assets/memes` and `assets/sounds` folders next to the executable.

## How it works

- **Face and gaze:** MediaPipe Face Landmarker gives head pose and eye direction.
- **Phone detection:** a MediaPipe object detector finds phones, and a CLIP check rejects look-alikes such as calculators.
- **Decision:** you count as "on your phone" only if your gaze points at a confirmed phone for 1.5 s.
