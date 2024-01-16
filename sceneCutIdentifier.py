import cv2

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """

    # open the video
    cap = cv2.VideoCapture(video_path)

    # ensure that that the video succeed to open
    if cap.isOpened():

        # min_delta = -20  # the minimal delta that define scene cut
        hists = []
        max_delta = 0  # the maximal delta between two consecutive frames
        max_indx = 0  # the indx of the first frame of the two consecutive frames where the scene cut occurred
        curr = 0  # the indx of the current frame
        prev = None  # the CDF of the prev frame, initialized as None

        # Read frames from the video
        while True:
            ret, frame = cap.read()

            if not ret:
                # Break the loop if there are no more frames
                break

            # convert to grayscale
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # gets the histogram
            curr_hist = cv2.calcHist([gray_scale], [0], None, [256], [0, 256])

            # calculate the CDF
            prev_hist = 0
            for i in range(len(curr_hist)):
                curr_hist[i] += prev_hist
                prev_hist = curr_hist[i]

            # calculate the diff between the curr and prev grayscale CDF
            delta = 0
            if prev is not None:
                for i in range(len(curr_hist)):
                    delta += abs(prev[i] - curr_hist[i])

                # if the curr diff greater than the max update the max
                if delta > max_delta:
                    max_delta = delta
                    max_indx = curr

            # update the previous frame anf the index
            prev = curr_hist
            hists.append(delta)
            curr += 1

        # if max_delta >= min_delta:
        return max_indx-1, max_indx










