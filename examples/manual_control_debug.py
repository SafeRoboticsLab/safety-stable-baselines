#!/usr/bin/env python3
"""
Manual control script for debugging the Circle environment.
Use keyboard to control the car and see the safety margin (g function) in real-time.
"""

import os
import sys
import numpy as np
import pygame
import time

# Add parent directory to path so we can import safety_sb3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_gymnasium.safety_envs.safety_circle_margin import make_env


def print_environment_info(env, obs, info):
    """Print detailed environment information for debugging."""
    print("\n" + "="*60)
    print("ENVIRONMENT DEBUG INFO")
    print("="*60)
    
    # Print observation info
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Print info dictionary
    print(f"\nInfo dictionary:")
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value} (shape: {value.shape})")
        else:
            print(f"  {key}: {value}")
    
    # Print margin function specifically
    if 'margin_g' in info:
        margin = info['margin_g']
        print(f"\n🛡️  SAFETY MARGIN (g function): {margin:.4f}")
        if margin > 0:
            print(f"   Status: ✅ SAFE (margin > 0)")
        else:
            print(f"   Status: ❌ UNSAFE (margin <= 0)")
        print(f"   Distance to safety boundary: {margin:.4f}m")
    else:
        print(f"\n❌ No margin_g found in info!")
    
    print("="*60)


def manual_control():
    """
    Manual control loop with keyboard input.
    """
    print("🚗 Manual Car Control - Circle Environment Debug")
    print("="*50)
    print("Controls (Differential Drive):")
    print("  W/S: Both wheels forward/backward (straight)")
    print("  A/D: Differential turning (left wheel slower/faster)")
    print("  Q/E: Left wheel backward/forward only")
    print("  Z/C: Right wheel backward/forward only") 
    print("  ESC: Quit")
    print("  R: Reset environment")
    print("  I: Print detailed info")
    print("="*50)
    
    # Create environment
    env = make_env(
        agent="Car", 
        level=2, 
        render_mode="human", 
        safety_clearance=0.0
    )
    
    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Manual Control (Focus this window for keyboard input)")
    clock = pygame.time.Clock()
    
    # Reset environment
    obs, info = env.reset()
    print("🎮 Environment initialized!")
    print_environment_info(env, obs, info)
    
    # Control parameters
    speed = 1.0  # Action magnitude
    running = True
    step_count = 0
    
    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get keyboard input
            keys = pygame.key.get_pressed()
            
            # Create action based on keyboard input
            # Action space: [left_wheel, right_wheel] velocities
            action = np.zeros(env.action_space.shape[0])
            
            # Basic movement (both wheels)
            if keys[pygame.K_w]:  # Forward
                action[0] = speed   # Left wheel forward
                action[1] = speed   # Right wheel forward
            elif keys[pygame.K_s]:  # Backward
                action[0] = -speed  # Left wheel backward
                action[1] = -speed  # Right wheel backward
            
            # Turning (differential drive)
            if keys[pygame.K_a]:  # Turn left (slow down left wheel)
                action[0] -= speed * 0.5  # Left wheel slower
                action[1] += speed * 0.5  # Right wheel faster
            elif keys[pygame.K_d]:  # Turn right (slow down right wheel)
                action[0] += speed * 0.5  # Left wheel faster
                action[1] -= speed * 0.5  # Right wheel slower
            
            # Individual wheel control
            if keys[pygame.K_q]:  # Left wheel backward
                action[0] = -speed
            elif keys[pygame.K_e]:  # Left wheel forward
                action[0] = speed
                
            if keys[pygame.K_z]:  # Right wheel backward
                action[1] = -speed
            elif keys[pygame.K_c]:  # Right wheel forward
                action[1] = speed
            
            # Special commands
            if keys[pygame.K_ESCAPE]:  # Quit (changed from Q to avoid conflict)
                running = False
                continue
            
            if keys[pygame.K_r]:  # Reset
                obs, info = env.reset()
                step_count = 0
                print("\n🔄 Environment reset!")
                print_environment_info(env, obs, info)
                continue
            
            if keys[pygame.K_i]:  # Print info
                print_environment_info(env, obs, info)
                time.sleep(0.5)  # Prevent spam
                continue
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print step information
            print(f"\n--- Step {step_count} ---")
            print(f"Action: [L:{action[0]:+.2f}, R:{action[1]:+.2f}] (Left/Right wheels)")
            print(f"Reward: {reward:.4f}")
            
            # Print safety margin prominently
            if 'margin_g' in info:
                margin = info['margin_g']
                status = "✅ SAFE" if margin > 0 else "❌ UNSAFE"
                print(f"🛡️  Safety Margin: {margin:+.4f} ({status})")
            
            # Print other useful info
            if 'cost' in info:
                print(f"💰 Cost: {info['cost']}")
            
            if 'goal_met' in info:
                print(f"🎯 Goal Met: {info['goal_met']}")
            
            # Check if episode ended
            if terminated:
                print(f"\n🛑 Episode TERMINATED!")
                if 'margin_g' in info and info['margin_g'] < 0:
                    print("   Reason: Safety violation (margin < 0)")
                print("   🔄 Auto-resetting environment...")
                obs, info = env.reset()
                step_count = 0
                time.sleep(1.0)  # Brief pause to read the message
                continue
            
            if truncated:
                print(f"\n⏰ Episode TRUNCATED (timeout)!")
                print("   🔄 Auto-resetting environment...")
                obs, info = env.reset()
                step_count = 0
                time.sleep(1.0)  # Brief pause to read the message
                continue
            
            # Render environment
            env.render()
            
            # Control loop speed
            clock.tick(20)  # 20 FPS
            time.sleep(0.05)  # Small delay for readability
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Manual control interrupted by Ctrl+C")
    
    finally:
        try:
            env.close()
        except Exception as e:
            # Ignore MuJoCo renderer cleanup errors
            if "get_current_context" not in str(e):
                print(f"Warning: Error closing environment: {e}")
        
        try:
            pygame.quit()
        except:
            pass  # Ignore pygame cleanup errors
        
        print(f"\n📊 Session Summary:")
        print(f"   Total steps: {step_count}")
        print(f"   Final margin: {info.get('margin_g', 'N/A')}")
        print(f"   Manual control session ended.")


def main():
    print("🚗 Circle Environment Manual Control & Debug Tool")
    print("This script lets you control the car manually while observing the safety margin.")
    print("Make sure you have pygame installed: pip install pygame")
    print()
    
    try:
        manual_control()
    except ImportError as e:
        if "pygame" in str(e):
            print("❌ pygame is required for keyboard input.")
            print("Install it with: pip install pygame")
        else:
            raise e
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e


if __name__ == "__main__":
    main()