import numpy as np
import matplotlib.pyplot as plt


def visualize_frames(clip, dxs, dys, t0=0, t0_len=1, exact_frames=None):
    """
    Visualize frames in clip, and superimpose markers given by dxs and dxy.

    Inputs:
    -------
    :param clip: VideoClip class from moviepy
        See moviepy VideoClip documentation for details
    :param dxs: (D, T) array
    :param dys: (D, T) array
    :param t0: int
        index of minimum frame to plot
    :param t0_len: int
        length of frames to plot
    :param exact_frames: None or list
        if list, plot frames in exact_frames.
            if t0_len > 1,  selects subset of frames to plot from exact_frames
            else plots all frames in exact_frames
        if None:
            plots frames in range(t0, t0 + t0_len)
    :return:
        makes plot
    """
    xlim, ylim = clip.size
    fps = clip.fps
    if np.ndim(dxs) == 1:
        num_markers = 1
        dxs = dxs[None, :]
        dys = dys[None, :]
    else:
        num_markers = dxs.shape[0]

    color_class = plt.cm.ScalarMappable(cmap="cool")
    colors = color_class.to_rgba(np.linspace(0, 1, num_markers))

    if exact_frames is None:
        exact_frames = range(t0, t0 + t0_len)
    else:
        if t0_len > 1:
            exact_frames = np.random.choice(exact_frames, t0_len)

    for frame_idx in exact_frames:
        print(frame_idx)
        frame_idx_sec = frame_idx / fps

        title_ = "Frame id {} @ time {:.2f} [sec]".format(frame_idx, frame_idx_sec)
        plt.figure(figsize=(10, 8))
        plt.imshow(clip.get_frame(frame_idx_sec))

        for part_idx in range(num_markers):
            title_ += "\n ({:.2f}, {:.2f})".format(
                dxs[part_idx, frame_idx], dys[part_idx, frame_idx]
            )
            plt.plot(
                dxs[part_idx, frame_idx],
                dys[part_idx, frame_idx],
                c=colors[part_idx],
                marker="o",
                ms=10,
            )
            plt.title(title_)
        plt.xlim([0, xlim])
        plt.ylim([ylim, 0])
        plt.tight_layout()
        plt.show()
    return


def visualize_frames_compare(clip, dxs, dys, dx, dy, t0=0, t0_len=1, exact_frames=None):
    """
    Visualize frames in clip, and superimpose markers given by dxs and dxy.

    Inputs:
    -------
    :param clip: VideoClip class from moviepy
        See moviepy VideoClip documentation for details
    :param dxs: (D, T) array
        marker coordinates plotted without facecolor
    :param dys: (D, T) array
        marker coordinates plotted without facecolor    
    :param dx: (D, T) array
        marker coordinates plotted with facecolor
    :param dy: (D, T) array
        marker coordinates plotted with facecolor
    
    :param t0: int
        index of minimum frame to plot
    :param t0_len: int
        length of frames to plot
    :param exact_frames: None or list
        if list, plot frames in exact_frames.
            if t0_len > 1,  selects subset of frames to plot from exact_frames
            else plots all frames in exact_frames
        if None:
            plots frames in range(t0, t0 + t0_len)
    :return:
        makes plot
    """
    xlim, ylim = clip.size
    fps = clip.fps
    if np.ndim(dxs) == 1:
        num_markers = 1
        dxs = dxs[None, :]
        dys = dys[None, :]
    else:
        num_markers = dxs.shape[0]

    color_class = plt.cm.ScalarMappable(cmap="cool")
    colors = color_class.to_rgba(np.linspace(0, 1, num_markers))

    if exact_frames is None:
        exact_frames = range(t0, t0 + t0_len)
    else:
        if t0_len > 1:
            exact_frames = np.random.choice(exact_frames, t0_len)

    for frame_idx in exact_frames:
        print(frame_idx)
        frame_idx_sec = frame_idx / fps

        title_ = "Frame id {} @ time {:.2f} [sec]".format(frame_idx, frame_idx_sec)
        plt.figure(figsize=(10, 8))
        plt.imshow(clip.get_frame(frame_idx_sec))

        for part_idx in range(num_markers):
            title_ += "\n ({:.2f}, {:.2f})  ({:.2f}, {:.2f})".format(
                dxs[part_idx, frame_idx], dys[part_idx, frame_idx],
                dx[part_idx, frame_idx], dy[part_idx, frame_idx]
            )
            plt.plot(
                dxs[part_idx, frame_idx],
                dys[part_idx, frame_idx],
                c=colors[part_idx],
                marker="o",
                markerfacecolor='none',
                ms=10,
            )
            plt.plot(
                dx[part_idx, frame_idx],
                dy[part_idx, frame_idx],
                c=colors[part_idx],
                marker="o",
                ms=10,
            )
            plt.title(title_)
        plt.xlim([0, xlim])
        plt.ylim([ylim, 0])
        plt.tight_layout()
        plt.show()
    return
