# RANSAC 원근 변환 근사 계산으로 나쁜 매칭 제거 (match_homography_accuracy.py)
import cv2, numpy as np
import pickle
import os
from PIL import ImageDraw, Image

def read_file(datadir, filename):
    if not os.path.exists(datadir):
        exit(-1)

    li = []
    with open(os.path.join(datadir, filename), 'rb') as fp:
        while True:
            try:
                data = pickle.load(fp)
            except EOFError:
                break
            li.append(data)

    return li


def set_cor_mosaic():   
    datadir = './match'
    p_in = np.asarray([list(i) for i in read_file(datadir, 'porto1.bin')])
    p_ref = np.asarray([list(i) for i in read_file(datadir, 'porto2.bin')])

    return p_in, p_ref


def save_points_image(img, points, resultdir, filename):
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    r = 3
    draw = ImageDraw.Draw(img)
    for point in points:
        draw.ellipse((point[0]-r, point[1]-r, point[0]+r, point[1]+r), fill=(255,0,0,0))
    img.save(os.path.join(resultdir, filename))


def find_cor_points():
    img1 = cv2.imread('./data/porto1.png')
    img2 = cv2.imread('./data/porto2.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB, BF-Hamming 로 knnMatch  ---①
    detector = cv2.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)

    # 모든 매칭점 그리기 ---④
    res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 매칭점으로 원근 변환 및 영역 표시 ---⑤
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
    # RANSAC으로 변환 행렬 근사 계산 ---⑥
    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape[:2]
    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    dst = cv2.perspectiveTransform(pts,mtrx)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # 정상치 매칭만 그리기 ---⑦
    matchesMask = mask.ravel().tolist()
    res2 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                        matchesMask = matchesMask,
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    list_kp1_all = [kp1[mat.queryIdx].pt for mat, mask in zip(matches, matchesMask)] 
    list_kp2_all = [kp2[mat.trainIdx].pt for mat, mask in zip(matches, matchesMask)]

    list_kp1 = []
    list_kp2 = []
    for mat, msk in zip(matches, matchesMask) :
        if msk :
            list_kp1.append(kp1[mat.queryIdx].pt)
            list_kp2.append(kp2[mat.trainIdx].pt)

    # 모든 매칭점과 정상치 비율 ---⑧
    accuracy=float(mask.sum()) / mask.size
    print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

    # 결과 저장
    datadir = './match'
    if not os.path.exists(datadir):
        os.makedirs(datadir) 
    with open(os.path.join(datadir, 'porto1.bin'), 'wb') as f:
        for item in list_kp1:
            pickle.dump(item, f)
    with open(os.path.join(datadir, 'porto2.bin'), 'wb') as f:
        for item in list_kp2:
            pickle.dump(item, f)

    resultdir = './results/match'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir) 
    cv2.imwrite(os.path.join(resultdir, 'match_all.png'), res1)
    cv2.imwrite(os.path.join(resultdir, 'match_inlier.png'), res2)


def main():
    resultdir = './results'
    find_cor_points()
    p_in, p_ref = set_cor_mosaic()
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')
    save_points_image(img_in, p_in, resultdir, "porto1_matched.png")
    save_points_image(img_ref, p_ref, resultdir, "porto2_matched.png")


if __name__ == '__main__':
    main()
    
