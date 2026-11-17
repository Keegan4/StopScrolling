StopScrolling
=============

Watches your webcam while you study. When you look at your phone for more than
1.5 seconds it plays an alarm and fills your screen with meme popups. They all
close as soon as you put the phone down. Everything runs on your computer and no
video is saved or sent anywhere.

Run it
------
Windows:  double-click StopScrolling.exe
          (Windows SmartScreen may warn about an unknown app: More info -> Run anyway)
macOS:    in Terminal, inside this folder:
              xattr -dr com.apple.quarantine .
              ./StopScrolling
          Allow camera access for Terminal when asked.
Linux:    ./StopScrolling   (needs an X11 desktop session for the popups, and the
          EGL/GLES libraries: sudo apt install libegl1 libgles2)

A camera window opens. Keys in that window:
  q / Esc   quit
  s         snooze alerts for 5 minutes (press again to resume)
  + / -     phone detector threshold
  [ / ]     how precisely you must be looking at the phone
  m         mirror the preview

Useful options (add after the program name):
  --no-preview      no camera window, just the console log (Ctrl+C to quit)
  --no-sound        popups only
  --no-popups       sound only
  --max-popups 15   fewer popups on screen at once
  --enter-s 3       look at the phone longer before alerts start
  --camera 1        use a different webcam
  --help            all options

Your own memes and sounds
-------------------------
Put images (.png .jpg .gif .webp) in assets/memes and sounds (.wav .ogg .mp3)
in assets/sounds. They are picked up the next time you start the app.
