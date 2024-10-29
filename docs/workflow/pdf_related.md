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
