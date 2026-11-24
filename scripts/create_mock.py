import logging
import cv2, numpy as np
from pathlib import Path
from crackseg.logging_utils import setup_logging

'''Simple utility scirpt for generating mock data'''

setup_logging()
logger = logging.getLogger(__name__)

root = Path("data/raw/mock")
(root / "images").mkdir(parents=True, exist_ok=True)
(root / "masks").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

#generating 50 "cracks" for mock purpose
for i in range(50):
    img = rng.integers(80, 160, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    pts = rng.integers(0, 256, (2, 2))
    cv2.line(mask, tuple(map(int, pts[0])), tuple(map(int, pts[1])), 255, thickness=3)
    img[mask > 0] = 30
    cv2.imwrite(str(root / "images" / f"{i:03d}.jpg"), img)
    cv2.imwrite(str(root / "masks" / f"{i:03d}.png"), mask)
    logger.debug("generated pair %d", i)

logger.info("generated 50 mock pairs in %s", root)


#obvious errors to be catched by direct file
cv2.imwrite(str(root / "masks" / "999.png"), np.zeros((256, 256), np.uint8))   #only mask
cv2.imwrite(str(root / "masks" / "000.png"), np.zeros((128, 128), np.uint8))   #wrong size