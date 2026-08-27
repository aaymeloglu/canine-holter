# Icon attribution

`icon.icns` is built from `icon-source/dog_face_3d.png`, the "Dog face" 3D
asset from Microsoft's [Fluent Emoji](https://github.com/microsoft/fluentui-emoji)
(`assets/Dog face/3D/dog_face_3d.png`), used under the MIT License:

> Copyright (c) Microsoft Corporation.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The `.icns` was produced from the PNG with macOS `sips` (resize) and
`iconutil` only; to regenerate:

```bash
mkdir icon.iconset
for sz in 16 32 128 256 512; do
  sips -z $sz $sz icon-source/dog_face_3d.png --out icon.iconset/icon_${sz}x${sz}.png
  sips -z $((sz*2)) $((sz*2)) icon-source/dog_face_3d.png --out icon.iconset/icon_${sz}x${sz}@2x.png
done
iconutil -c icns icon.iconset -o icon.icns
```

`icon.ico` (Windows) is the same PNG, resized with Pillow only; to regenerate:

```bash
python -c "from PIL import Image; Image.open('icon-source/dog_face_3d.png').convert('RGBA').save('icon.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
```
