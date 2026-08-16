/**
 * Custom High-Performance Neon Hand Landmark Canvas Renderer for SignSense AI
 */

export const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index Finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle Finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring Finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

export const FINGERTIP_INDICES = [4, 8, 12, 16, 20];

/**
 * Draws 21 hand landmarks and skeleton bones on HTML5 canvas with neon glow effects.
 * 
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
 * @param {Array<{x: number, y: number, z: number}>} landmarks - Normalized (0..1) 21 hand landmarks
 * @param {number} width - Canvas width in pixels
 * @param {number} height - Canvas height in pixels
 * @param {boolean} isMirrored - Whether canvas output is horizontally mirrored
 */
export function drawNeonHandLandmarks(ctx, landmarks, width, height, isMirrored = true) {
  if (!landmarks || landmarks.length !== 21) return;

  ctx.save();

  // Convert normalized landmark (0..1) to canvas pixel coordinates
  const getPixelCoord = (lm) => {
    const x = isMirrored ? (1 - lm.x) * width : lm.x * width;
    const y = lm.y * height;
    return { x, y, z: lm.z };
  };

  const pixelLandmarks = landmarks.map(getPixelCoord);

  // 1. Draw Skeleton Bone Connections
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#00f0ff'; // Neon Cyan
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 8;

  HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
    const p1 = pixelLandmarks[startIdx];
    const p2 = pixelLandmarks[endIdx];

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  });

  // 2. Draw Landmark Joint Nodes
  pixelLandmarks.forEach((pt, idx) => {
    const isTip = FINGERTIP_INDICES.includes(idx);
    const isWrist = idx === 0;

    ctx.beginPath();
    
    if (isTip) {
      // Fingertip: Neon Emerald with Outer Pulse Circle
      ctx.arc(pt.x, pt.y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#00ff9d';
      ctx.shadowColor = '#00ff9d';
      ctx.shadowBlur = 12;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 11, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(0, 255, 157, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (isWrist) {
      // Wrist Anchor: Pure White Node
      ctx.arc(pt.x, pt.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 14;
      ctx.fill();
    } else {
      // Internal Joints: Electric Purple Core with Cyan Border
      ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#7000ff';
      ctx.shadowColor = '#7000ff';
      ctx.shadowBlur = 6;
      ctx.fill();

      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });

  ctx.restore();
}
