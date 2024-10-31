# Разрешение и DPI изображений
- Тезисы: разрешение, dpi, их смысл
- Что такое 1080p
  - О чем говорит разрешение
  - пример про ютуб
- Что такое DPI
  - Связь с физическими размерами
  - пример из матплотлиб
  - пример из pdf2image.
- Как влияет на GPT?
  - картинки, примеры галюцинаций

**Разрешение = количество пикселей**. Оно выражается через произведение сторон `1920x1080`. Изображения с высоким расширением передают больше информации. 

**Стандартные разрешения:** 

| Youtube | Стороны   |
| ------- | --------- |
| 144p    | 256×144   |
| 270p    | 480×270   |
| 360p    | 640×360   |
| 540p    | 960×540   |
| 720p    | 1280×720  |
| 1080p   | 1920×1080 |


В этой таблице два правила: 
- 4/3 = Новое/Старое
- 16:9 - стандартное соотношение сторон.  

Поэтому в разрешении можно указывать только одно число. Обычно это высота.

На Youtube качество измеряется в 1080p, 720p, и тд. Это число показывает, сколько пикселей информации приходит на устройство. Затем эти пиксели нужно "натянуть" на экран. Рассмотрим пример, когда разрешение монитора 1920x1080:
- Качество 1080p: каждый пиксель отображается 1 в 1. Картинка четкая.
```
1080p on 1080p display:
Video pixel:    Monitor pixel:
[*]         →   [#]
```
- Качество 360p: на каждый пиксель изображения приходится 3x3=9 пикселей экрана (по двум осям картинки).
```
360p on 1080p display:
Video pixel:    Monitor pixels:
[*]         →   [# # #]
                [# # #]
                [# # #]
```

Разрешение не измеряет качество напрямую. Оно отражает количество информации. 

**DPI - Dots Per Inch**. Показывает количество пикселей в одном дюйме изображения.
















RESOLUTIONS & DPI BASICS:
1. Resolution = total pixels (1920×1080)
2. DPI = dots per inch (density of pixels)
3. Physical size = resolution ÷ DPI

DISPLAY:
- Common resolutions: 360p, 720p, 1080p
- Monitor has fixed:
  - Physical size (e.g., 24")
  - Native resolution (e.g., 1920×1080)
  - Resulting DPI

PDF SPECIFICS:
1. Contains multiple types of content:
   - Vector (text, shapes) - resolution independent
   - Raster (images) - fixed resolution
2. Has physical dimensions (stored in points)
   - 1 point = 1/72 inch
   - A4 = 595×842 points ≈ 8.27"×11.69"
   - Letter = 612×792 points = 8.5"×11"

PDF TO IMAGE CONVERSION:
```python
# pdf2image example
images_72 = convert_from_path('doc.pdf', dpi=72)
# 8.5" × 72 = 612 pixels wide
# 11" × 72 = 792 pixels tall

images_300 = convert_from_path('doc.pdf', dpi=300)
# 8.5" × 300 = 2550 pixels wide
# 11" × 300 = 3300 pixels tall
```

SCALING EXAMPLES:
1. Upscaling (more pixels):
```
[A][B]    →    [A][A/B][B]
[C][D]         [A/C][X][B/D]
               [C][C/D][D]
```

2. Downscaling (fewer pixels):
```
[A][B][C]      [X][Y]
[D][E][F]  →   [Z][W]
[G][H][I]
Where X = average(A,B,D,E)
```

YOUTUBE QUALITY:
- When changing resolution:
  - Changes number of pixels
  - Monitor DPI stays same
  - Pixels get stretched/compressed
- 720p on 1080p display:
  - 1 video pixel ≈ 2.25 monitor pixels
  - Requires interpolation

TYPICAL DPI VALUES:
- Web: 72-96 DPI
- Print: 300+ DPI
- Phone screens: 300-500+ DPI
- PowerPoint: 144 DPI (internal)

KEY POINTS:
1. PDFs mix vector and raster content
2. Monitor DPI is fixed hardware property
3. Changing video resolution affects pixels, not DPI
4. PDF conversion DPI determines output resolution
5. Vector content (like PDF text) is resolution independent
6. Physical size relationships:
   - More DPI = smaller physical size (same pixels)
   - More pixels = larger physical size (same DPI)

Is there anything specific you'd like me to expand on further?
