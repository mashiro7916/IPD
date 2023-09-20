from MOD.correspond_points import CorrespondPoints
import os
import cv2


def eliminate_duplicates(folder):
    key_points = 256
    CP = CorrespondPoints('superpoint', key_points)
    folder = folder
    input_path = os.path.join('/media/jacky72503/data/IPD/detect_result', folder)
    filenames = os.listdir(input_path)
    preimg = None
    curimg = None
    results = []
    for f in filenames:
        img = cv2.imread(os.path.join(input_path, f))
        if preimg is None:
            preimg = img
            continue
        elif curimg is None:
            curimg = img
        feats0, feats1, matches = CP.getmatch(preimg, curimg)
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        result = curimg.copy()
        # result = CP.vis(m_kpts0, m_kpts1, matches, result)
        # cv2.imshow('r', result)
        # cv2.waitKey(0)
        if len(m_kpts1) < (key_points / 4):
            exist = False
            for r in results:
                f0, f1, m = CP.getmatch(r, result)
                k0, k1, m = f0['keypoints'], f1['keypoints'], m['matches']
                m0, m1 = k0[m[..., 0]], k1[m[..., 1]]
                print('m1:',len(m1))
                if len(m1) > (key_points / 8):
                    exist = True
                    break
            if not exist:
                print('new object')
                results.append(result)

        preimg = curimg
        curimg = None
    # for r in results:
    #     cv2.imshow('r', r)
    #     cv2.waitKey(0)
    return results


# results = eliminate_duplicates('004')
# print(len(results))
# for r in results:
#     cv2.imshow('r',r)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
