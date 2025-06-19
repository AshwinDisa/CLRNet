import cv2
import os
import os.path as osp


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_lanes(img, lanes, show=False, out_file=None, width=4, video=False, fps=None, infer_time=None):
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
            # print(f'({x}, {y})')
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
            # print(f'Drawing lane {idx} from {xys[i - 1]} to {xys[i]}')

    if show and not video:
        cv2.imshow('view', img)
        cv2.waitKey()

    if show and video:
        if fps is not None:
            cv2.putText(img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
        if infer_time is not None:
            cv2.putText(img, f"Infer Time: {(infer_time * 1000):.2f} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)


        cv2.imshow('view', img)
        cv2.waitKey(1)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)