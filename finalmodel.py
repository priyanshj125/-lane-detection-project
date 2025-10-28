

import cv2
import numpy as np
import argparse
import sys

# ---------- Utility functions ----------

def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, left_fit, right_fit, ploty):
    # Generate points to draw the lane area
    left_pts = np.array([np.transpose(np.vstack([left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2], ploty]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2], ploty])) )])
    pts = np.hstack((left_pts, right_pts))
    lane_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.fillPoly(lane_img, np.int_([pts]), (0,255, 0))
    return lane_img


def perspective_transform(img):
    # Define source and destination points for a typical 1280x720-ish frame
    h, w = img.shape[:2]
    # These source coordinates work for many forward-facing car videos; you may need to tune.
    src = np.float32([
        [w*0.43, h*0.65],
        [w*0.58, h*0.65],
        [w*0.92, h*0.95],
        [w*0.15, h*0.95],
    ])
    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.8, h],
        [w*0.2, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def color_threshold(img):
    # Convert to HLS and threshold the S channel (works well for lane yellow/white)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    # Thresholds -- tweak if necessary
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1

    # Also detect white lanes via lightness
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1

    combined = np.zeros_like(s_channel)
    combined[(s_binary == 1) | (l_binary == 1)] = 255
    return combined


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Output image for visualization
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of left and right halves
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    if len(leftx) == 0 or len(rightx) == 0:
        return None, None, out_img
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    return left_fit, right_fit, ploty


def measure_curvature_real(left_fit, right_fit, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension (approx lane width)
    y_eval = np.max(ploty)

    # Convert polynomial coefficients to real world
    left_fit_cr = np.array([left_fit[0]*xm_per_pix/(ym_per_pix**2), left_fit[1]*xm_per_pix/ym_per_pix, left_fit[2]*xm_per_pix])
    right_fit_cr = np.array([right_fit[0]*xm_per_pix/(ym_per_pix**2), right_fit[1]*xm_per_pix/ym_per_pix, right_fit[2]*xm_per_pix])

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def process_frame(frame):
    orig = frame.copy()
    h, w = frame.shape[:2]

    # 1) Color thresholding to highlight lane lines
    color_bin = color_threshold(frame)

    # 2) Perspective transform
    warped_color, M, Minv = perspective_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Apply color threshold on warped image as well
    warped_color_bgr = cv2.cvtColor(warped_color, cv2.COLOR_RGB2BGR)
    warped_bin = color_threshold(warped_color_bgr)

    # 3) Convert to binary and run sliding windows
    if len(warped_bin.shape) == 3:
        warped_bin_gray = cv2.cvtColor(warped_bin, cv2.COLOR_BGR2GRAY)
    else:
        warped_bin_gray = warped_bin

    # ensure binary is 0/255
    _, warped_binary = cv2.threshold(warped_bin_gray, 1, 255, cv2.THRESH_BINARY)

    left_fit, right_fit, ploty = fit_polynomial(warped_binary)
    if left_fit is None or right_fit is None:
        # fallback: return original frame if detection fails
        return orig

    # Create lane drawing
    lane_warp = draw_lines(orig, left_fit, right_fit, ploty)

    # Warp lane back to original perspective
    h, w = orig.shape[:2]
    lane_unwarped = cv2.warpPerspective(lane_warp, Minv, (w, h))

    # Combine with original
    result = cv2.addWeighted(orig, 1.0, lane_unwarped, 0.5, 0)

    # Compute curvature and vehicle position
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    avg_curv = (left_curverad + right_curverad) / 2.0

    # Vehicle center offset
    # Evaluate polynomials at bottom of image (max y)
    y_eval = np.max(ploty)
    left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x_bottom + right_x_bottom) / 2.0
    vehicle_center = orig.shape[1] / 2.0
    xm_per_pix = 3.7/700
    center_offset_m = (vehicle_center - lane_center) * xm_per_pix

    # Overlay text
    cv2.putText(result, f"Radius of Curvature: {avg_curv:.0f} m", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(result, f"Center Offset: {center_offset_m:.2f} m", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    return result


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description='Lane detection on video')
    parser.add_argument('--input', type=str, required=True, help='Input video file path or webcam index (0,1...)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file path')
    args = parser.parse_args()

    # Determine if input is webcam index
    use_cam = False
    try:
        cam_index = int(args.input)
        use_cam = True
    except Exception:
        use_cam = False

    if use_cam:
        cap = cv2.VideoCapture(cam_index)
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print('Error opening video stream or file')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print('Processing... press q to quit early')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out.write(processed)

        # Show live (optional)
        cv2.imshow('Lane Detection', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Done. Output saved to', args.output)


if __name__ == '__main__':
    main()
