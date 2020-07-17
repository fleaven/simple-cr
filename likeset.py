#形近字字符集
import os
import shutil

wholesetdir="../ocr-dataset/hwdb/train/"
likesetdir="../ocr-dataset/hwdb/minitrain/"
likeset="日目百白旦赛塞寒桌卓焯倬淖棹琸晫啅悼绰逴婥直值置植殖者都赌堵睹绪猪诸煮署督暑躇真慎填"

for ch in likeset:
    if (os.path.exists(wholesetdir+ch)):
        shutil.copytree(wholesetdir+ch, likesetdir+ch)
