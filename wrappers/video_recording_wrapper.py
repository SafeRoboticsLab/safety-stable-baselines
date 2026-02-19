"""
Video Recording Wrapper for Safety Gymnasium Environments

This wrapper records episodes and saves them as video files after each episode completes.
Uses gymnasium's video recorder functionality with custom episode naming.
"""

import os
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import Optional, Callable
import imageio
from gymnasium import logger

class VideoRecordingWrapper(gym.Wrapper):
    """
    Environment wrapper that records episodes and saves them as video files.
    
    Features:
    - Records each episode automatically
    - Saves videos with episode number and statistics in filename
    - Supports custom video directory and naming
    - Optional frame rate control
    - Includes episode metadata in video filename
    
    Args:
        env: The environment to wrap
        video_folder: Directory to save videos (default: "./videos")
        name_prefix: Prefix for video filenames (default: "episode")
        fps: Frames per second for video (default: 30)
        record_every_n_episodes: Record only every N episodes (default: 1 = record all)
        include_stats_in_name: Include episode stats in filename (default: True)
        camera_name: Camera to use for recording (default: None, uses env default)
                     Options: 'vision', 'track', 'fixednear', 'fixedfar', 'human'
        camera_id: Camera ID to use (alternative to camera_name)
    """
    
    def __init__(
        self,
        env: gym.Env,
        video_folder: str = "./videos",
        name_prefix: str = "episode",
        fps: int = 30,
        record_every_n_episodes: int = 1,
        include_stats_in_name: bool = True,
    ):
        super().__init__(env)
        
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.fps = fps
        self.record_every_n_episodes = record_every_n_episodes
        self.include_stats_in_name = include_stats_in_name
        
        # Create video directory if it doesn't exist
        os.makedirs(self.video_folder, exist_ok=True)
        
        # Episode tracking
        self.episode_count = 0
        self.is_recording = False
        self.frames = []
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.episode_steps = 0
        
        print(f"   VideoRecordingWrapper initialized:")
        print(f"   Video folder: {self.video_folder}")
        print(f"   FPS: {self.fps}")
        print(f"   Recording every {self.record_every_n_episodes} episode(s)")
    
    def reset(self, **kwargs):
        """Reset environment and start recording if appropriate."""
        obs, info = self.env.reset(**kwargs)
        
        # Increment episode counter
        self.episode_count += 1
        
        # Reset episode statistics
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        self.episode_steps = 0
        
        # Determine if we should record this episode
        self.is_recording = (self.episode_count % self.record_every_n_episodes == 0)
        
        if self.is_recording:
            # Clear previous frames
            self.frames = []
            
            # Capture initial frame
            frame = self._get_frame()
            if frame is not None:
                self.frames.append(frame)
            
            print(f"Started recording Episode {self.episode_count}")
        
        return obs, info

    def soft_reset(self, **kwargs):
        """A soft reset that does not reset the environment state."""
        if hasattr(self.env, 'soft_reset'):
            self._elapsed_steps = 0
            self.frames = [] # drop the first frame
            return self.env.soft_reset(**kwargs)
        else:
            logger.warn(
                'The environment does not support soft reset. '
                'Falling back to hard reset.',
            )
            return self.reset(**kwargs)
    
    def step(self, action):
        """Step environment and record frame if recording."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_steps += 1
        
        if 'cost' in info:
            self.episode_cost += info['cost']
        
        # Record frame if we're recording this episode
        if self.is_recording:
            frame = self._get_frame()
            if frame is not None:
                self.frames.append(frame)
        
        # If episode is done and we were recording, save the video
        if (terminated or truncated) and self.is_recording:
            self._save_video(terminated, truncated)
        
        return obs, reward, terminated, truncated, info
    
    def _get_frame(self):
        """Get current frame from environment."""
        try:
            frame = self.env.render()
            
            # Handle different return types
            if isinstance(frame, np.ndarray):
                # Ensure correct shape (H, W, C)
                if len(frame.shape) == 2:
                    # Grayscale - convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                elif len(frame.shape) == 3 and frame.shape[0] in [3, 4]:
                    # Channel first (C, H, W) - transpose to (H, W, C)
                    frame = np.transpose(frame, (1, 2, 0))
                
                # Ensure uint8 type
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                return frame
            
            return None
            
        except Exception as e:
            if len(self.frames) == 0:  # Only warn on first failure
                print(f" Warning: Could not capture frame: {e}")
            return None
    
    def _save_video(self, terminated: bool, truncated: bool):
        """Save recorded frames as a video file."""
        if not self.frames:
            print(f" No frames to save for Episode {self.episode_count}")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.include_stats_in_name:
            # Include statistics in filename
            reason = "terminated" if terminated else "truncated"
            filename = (
                f"{self.name_prefix}_ep{self.episode_count:04d}_"
                f"r{self.episode_reward:.1f}_c{self.episode_cost:.1f}_"
                f"s{self.episode_steps}_{reason}_{timestamp}.mp4"
            )
        else:
            # Simple filename
            filename = f"{self.name_prefix}_ep{self.episode_count:04d}_{timestamp}.mp4"
        
        video_path = os.path.join(self.video_folder, filename)
        
        try:
            # Save video using imageio
            imageio.mimsave(
                video_path,
                self.frames,
                fps=self.fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            
            print(f"   Saved video: {filename}")
            print(f"   Frames: {len(self.frames)}, Duration: {len(self.frames)/self.fps:.2f}s")
            print(f"   Reward: {self.episode_reward:.3f}, Cost: {self.episode_cost:.3f}, Steps: {self.episode_steps}")
            
        except Exception as e:
            print(f"   Error saving video: {e}")
            print(f"   Attempted to save {len(self.frames)} frames to {video_path}")
        
        # Clear frames to free memory
        self.frames = []
    
    def close(self):
        """Close environment and clean up."""
        # Save any remaining frames if recording was interrupted
        if self.is_recording and self.frames:
            print("Saving incomplete episode recording...")
            self._save_video(terminated=False, truncated=True)
        
        print(f"\nVideoRecordingWrapper Statistics:")
        print(f"   Total episodes: {self.episode_count}")
        print(f"   Videos saved in: {self.video_folder}")
        
        return super().close()