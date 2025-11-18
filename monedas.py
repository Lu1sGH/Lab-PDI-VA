import cv2
import numpy as np

rangos = {
    "10":  (5369, 5769),   
    "5":   (4491, 4642),
    "2":   (3612, 3793),
    "1":   (2910, 2970),
    "50c": (1950, 1999)
}

def obtener_denominacion(area):
    for denom, (a_min, a_max) in rangos.items():
        if a_min <= area <= a_max:
            return denom
    return "?"   # si no cae en ningún rango

scale = 0.4
img = cv2.imread("Fotos/Test/monedas3.jpeg", 0)
img = cv2.resize(img, None, fx=scale, fy=scale)

# Imagen original a color para dibujar sobre ella el resultado final
img_color = cv2.imread("Fotos/Test/monedas3.jpeg")
img_color = cv2.resize(img_color, None, fx=scale, fy=scale)

# Canny + Closing
borders = cv2.Canny(img, 100, 300)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
closing = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)
min_area = 500
clean = np.zeros_like(closing) # Matriz de puros ceros del tamaño de closing

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        clean[labels == i] = 255
        
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# center, radius = cv2.minEnclosingCircle(contours)

# Rellenar contornos
cv2.drawContours(clean, contours, -1, 255, -1)

dinero = 0

# Dibujar resultados sobre la imagen original a color
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    denom = obtener_denominacion(area)

    if denom != "?":
        if denom == "50c":
            dinero += 0.5
        else:
            dinero += int(denom)
    # Dibujar contorno
    cv2.drawContours(img_color, [cnt], -1, (255, 255, 255), 2)

    # Centroide para colocar texto
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = cnt[0][0]

    # Escribir el área y la denominación
    texto = f"{area}  -> {denom}"
    cv2.putText(img_color, texto, (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print(f"Moneda {i+1}: área = {area}, denominación = {denom}")

print(f"\nTotal monedas detectadas: {len(contours)}")
print(f"\nTotal de dinero: {dinero}")
        
cv2.imshow("Imagen", img_color)
cv2.waitKey()
cv2.destroyAllWindows()