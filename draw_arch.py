import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(28, 22))
ax.set_xlim(0, 28)
ax.set_ylim(0, 22)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=9, fontweight='bold', alpha=0.9, rounded=True, linewidth=1.5):
    if rounded:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.3",
                             facecolor=color, edgecolor='white', linewidth=linewidth, alpha=alpha)
    else:
        box = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                             facecolor=color, edgecolor='white', linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#aaaaaa', style='->', lw=1.5, connectionstyle="arc3,rad=0"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle=connectionstyle))

c_title = '#e94560'
c_backbone = '#0f3460'
c_stem = '#16213e'
c_stage = '#1a1a40'
c_encoder = '#533483'
c_aifi = '#7952b3'
c_fpn = '#e94560'
c_pan = '#f39c12'
c_decoder = '#00b894'
c_head = '#0984e3'
c_loss = '#d63031'
c_input = '#00cec9'
c_output = '#fdcb6e'
c_feat = '#6c5ce7'

ax.text(14, 21.3, 'DEIMv2 Network Architecture', ha='center', va='center',
        fontsize=24, fontweight='bold', color=c_title)
ax.text(14, 20.6, 'HGNetv2-B0 + HybridEncoder (dfine) + DEIMTransformer | PEST Detection Config',
        ha='center', va='center', fontsize=12, color='#888888', style='italic')

draw_box(ax, 11.5, 19.4, 5, 0.8, 'Input Image\n640×640×3', c_input, fontsize=10)

bx, by = 5.5, 16.5
draw_box(ax, bx, by, 17, 2.5, '', c_backbone, alpha=0.3, rounded=True)
ax.text(bx+0.3, by+2.25, 'HGNetv2-B0 Backbone', ha='left', va='top', fontsize=13,
        fontweight='bold', color='#4fc3f7')

stem_x = 6.5
draw_box(ax, stem_x, by+0.3, 2.2, 1.9, 'Stem\nConv(3→16)\nConv(16→16)', c_stem, fontsize=8)
s1_x = 9.2
draw_box(ax, s1_x, by+0.5, 2.2, 1.5, 'Stage1\nC2f×1\n64ch /4', c_stage, fontsize=8)
s2_x = 11.9
draw_box(ax, s2_x, by+0.5, 2.2, 1.5, 'Stage2\nC2f×1 ↓\n256ch /8', c_stage, fontsize=8)
s3_x = 14.6
draw_box(ax, s3_x, by+0.3, 2.4, 1.7, 'Stage3\nLightBottle×2 ↓\n512ch /16', c_stage, fontsize=8)
s4_x = 17.5
draw_box(ax, s4_x, by+0.3, 2.4, 1.7, 'Stage4\nLightBottle×1 ↓\n1024ch /32', c_stage, fontsize=8)
s5_x = 20.4
draw_box(ax, s5_x, by+0.5, 1.8, 1.5, 'SPPF\n1024ch', c_stage, fontsize=8)

draw_arrow(ax, 14, 19.4, stem_x+1.1, by+2.2, '#4fc3f7', lw=2)
draw_arrow(ax, stem_x+2.2, by+1.25, s1_x, by+1.25, '#4fc3f7', lw=1.5)
draw_arrow(ax, s1_x+2.2, by+1.25, s2_x, by+1.25, '#4fc3f7', lw=1.5)
draw_arrow(ax, s2_x+2.2, by+1.25, s3_x, by+1.15, '#4fc3f7', lw=1.5)
draw_arrow(ax, s3_x+2.4, by+1.15, s4_x, by+1.15, '#4fc3f7', lw=1.5)
draw_arrow(ax, s4_x+2.4, by+1.15, s5_x, by+1.25, '#4fc3f7', lw=1.5)

feat_y = 15.3
draw_box(ax, s2_x+0.1, feat_y, 2.0, 0.65, 'P2: 256ch\nstride=8', c_feat, fontsize=7.5)
draw_box(ax, s3_x+0.1, feat_y-0.85, 2.0, 0.65, 'P3: 512ch\nstride=16', c_feat, fontsize=7.5)
draw_box(ax, s4_x+0.1, feat_y-1.7, 2.0, 0.65, 'P4: 1024ch\nstride=32', c_feat, fontsize=7.5)

draw_arrow(ax, s2_x+1.1, by+0.5, s2_x+1.1, feat_y+0.65, c_feat, lw=1.2)
draw_arrow(ax, s3_x+1.2, by+0.3, s3_x+1.2, feat_y-0.2, c_feat, lw=1.2)
draw_arrow(ax, s4_x+1.2, by+0.3, s4_x+1.2, feat_y-1.05, c_feat, lw=1.2)

enc_x, enc_y = 5.5, 10.0
draw_box(ax, enc_x, enc_y, 17, 4.8, '', c_encoder, alpha=0.25, rounded=True)
ax.text(enc_x+0.3, enc_y+4.55, 'HybridEncoder (version=dfine)', ha='left', va='top', fontsize=13,
        fontweight='bold', color='#f39c12')

proj_y = enc_y + 3.5
draw_box(ax, 7.0, proj_y, 2.8, 0.7, 'InputProj Conv1x1\n256→128', '#2d3436', fontsize=8)
draw_box(ax, 10.5, proj_y, 2.8, 0.7, 'InputProj Conv1x1\n512→128', '#2d3436', fontsize=8)
draw_box(ax, 14.0, proj_y, 2.8, 0.7, 'InputProj Conv1x1\n1024→128', '#2d3436', fontsize=8)

aifi_x, aifi_y = 18.0, enc_y + 2.8
draw_box(ax, aifi_x, aifi_y, 3.8, 1.8, 'AIFI\nTransformerEncoder\n(1 layer)\nhidden=128, ffn=512', c_aifi, fontsize=8)

draw_arrow(ax, s3_x+1.1, feat_y-0.85, 11.9, proj_y+0.7, '#aaa', lw=1)
draw_arrow(ax, s4_x+1.1, feat_y-1.7, 15.4, proj_y+0.7, '#aaa', lw=1)
draw_arrow(ax, 14.0+1.4, proj_y, aifi_x+1.9, aifi_y+1.8, '#bbb', lw=1.2)

fpn_y = enc_y + 1.8
lat_x = 7.0
draw_box(ax, lat_x, fpn_y-0.1, 2.2, 0.6, 'LateralConv\nConv1x1', '#636e72', fontsize=7.5)
fuse_x = 10.0
draw_box(ax, fuse_x, fpn_y-0.35, 3.2, 1.1, 'FPN Fuse Block\nRepNCSPELAN4\n(exp=0.34, depth=0.5)', c_fpn, fontsize=7.5)
pan_fuse_x = 10.0
pan_y = enc_y + 0.3
draw_box(ax, pan_fuse_x, pan_y-0.35, 3.2, 1.1, 'PAN Fuse Block\nRepNCSPELAN4\n(exp=0.34, depth=0.5)', c_pan, fontsize=7.5)
scdown_x = 7.0
draw_box(ax, scdown_x, pan_y-0.1, 2.2, 0.6, 'SCDown\nDWConv↓', '#636e72', fontsize=7.5)

out_y = enc_y - 0.8
draw_box(ax, fuse_x+0.1, out_y, 3.0, 0.6, 'Out: 128ch /8', c_output, fontsize=8)
out2_y = out_y - 0.9
draw_box(ax, pan_fuse_x+0.1, out2_y, 3.0, 0.6, 'Out: 128ch /16', c_output, fontsize=8)
out3_y = out2_y - 0.9
draw_box(ax, 17.5, out3_y, 3.0, 0.6, 'Out: 128ch /32', c_output, fontsize=8)

draw_arrow(ax, aifi_x+1.9, aifi_y, 17.5+1.5, out3_y+0.6, '#bbb', lw=1.2)
draw_arrow(ax, lat_x+1.1, fpn_y-0.1, fuse_x+1.6, fpn_y+0.75, '#aaa', lw=1)
draw_arrow(ax, scdown_x+1.1, pan_y-0.1, pan_fuse_x+1.6, pan_y+0.4, '#aaa', lw=1)
draw_arrow(ax, fuse_x+1.6, fpn_y-0.35, fuse_x+1.5, out_y+0.6, c_fpn, lw=1.2)
draw_arrow(ax, pan_fuse_x+1.6, pan_y-0.35, pan_fuse_x+1.5, out2_y+0.6, c_pan, lw=1.2)

dec_x, dec_y = 5.5, 4.2
draw_box(ax, dec_x, dec_y, 17, 4.3, '', c_decoder, alpha=0.25, rounded=True)
ax.text(dec_x+0.3, dec_y+4.05, 'DEIMTransformer Decoder (3 Layers)', ha='left', va='top', fontsize=13,
        fontweight='bold', color='#00b894')

query_x = 6.2
draw_box(ax, query_x, dec_y+2.6, 2.8, 1.2, 'Learnable Queries\n(300 queries\ndim=128)', '#2d3436', fontsize=8)
ref_x = 9.5
draw_box(ax, ref_x, dec_y+2.6, 2.6, 1.2, 'Reference Points\n(init & refine)\nsigmoid activation', '#2d3436', fontsize=8)

layer_w = 2.3
layer_h = 2.8
for i in range(3):
    lx = 12.8 + i * 2.8
    ly = dec_y + 0.6
    colors_layer = ['#00b894', '#00cec9', '#81ecec']
    draw_box(ax, lx, ly, layer_w, layer_h,
             f'DecLayer {i+1}\n─────────\nSelf-Attn\n(MHA)\nCross-Attn\n(MSDeformable)\nSwiGLU FFN',
             colors_layer[i], fontsize=7, alpha=0.85)
    if i == 0:
        draw_arrow(ax, query_x+1.4, dec_y+2.6, lx+layer_w/2, ly+layer_h, '#81ecec', lw=1.5)
        draw_arrow(ax, ref_x+1.3, dec_y+2.6, lx+layer_w+0.3, ly+layer_h*0.7, '#aaa', lw=1)
    else:
        prev_lx = 12.8 + (i-1) * 2.8
        draw_arrow(ax, prev_lx+layer_w, ly+layer_h*0.5, lx, ly+layer_h*0.5, '#81ecec', lw=1.5)

head_x = 21.5
draw_box(ax, head_x, dec_y+2.2, 2.8, 1.8, 'Detection Heads\n─────────────\n• Score Head\n• BBox Head (FDR)\n• QueryPos Head\n• LQE (Quality)', c_head, fontsize=7.5)
draw_arrow(ax, 12.8+2*2.8+layer_w, dec_y+0.6+layer_h*0.5, head_x, dec_y+3.1, c_head, lw=1.8)

loss_x, loss_y = 5.5, 0.3
draw_box(ax, loss_x, loss_y, 17, 1.6, '', c_loss, alpha=0.2, rounded=True)
ax.text(loss_x+0.3, loss_y+1.4, 'DEIMCriterion Loss', ha='left', va='top', fontsize=12,
        fontweight='bold', color=c_loss)
losses = [
    ('loss_mal', '1.0', 7.0),
    ('loss_bbox', '5.0', 10.0),
    ('loss_giou', '2.0', 13.0),
    ('loss_fgl', '0.15', 16.0),
    ('loss_ddf', '1.5', 19.0),
]
for name, weight, lx in losses:
    draw_box(ax, lx, loss_y+0.15, 2.6, 1.1, f'{name}\nw={weight}', '#c0392b', fontsize=8, alpha=0.8)

draw_arrow(ax, head_x+1.4, dec_y+2.2, 19.0, loss_y+1.6, c_loss, lw=1.5)

feat_labels = [
    ('P2\n/8', 8.0, out_y+0.1),
    ('P3\n/16', 11.0, out2_y+0.1),
    ('P4\n/32', 18.0, out3_y+0.1),
]
for label, fx, fy in feat_labels:
    ax.annotate('', xy=(12.8+0.3, dec_y+0.6+layer_h*0.3), xytext=(fx, fy),
                arrowprops=dict(arrowstyle='->', color=c_feat, lw=1.2,
                               connectionstyle="arc3,rad=-0.15", linestyle='dashed'))

legend_elements = [
    mpatches.Patch(facecolor=c_backbone, edgecolor='white', label='Backbone (HGNetv2-B0)'),
    mpatches.Patch(facecolor=c_encoder, edgecolor='white', label='HybridEncoder'),
    mpatches.Patch(facecolor=c_aifi, edgecolor='white', label='AIFI (Transformer Encoder)'),
    mpatches.Patch(facecolor=c_fpn, edgecolor='white', label='FPN (Top-down)'),
    mpatches.Patch(facecolor=c_pan, edgecolor='white', label='PAN (Bottom-up)'),
    mpatches.Patch(facecolor=c_decoder, edgecolor='white', label='DEIMTransformer Decoder'),
    mpatches.Patch(facecolor=c_head, edgecolor='white', label='Detection Heads'),
    mpatches.Patch(facecolor=c_loss, edgecolor='white', label='Loss Functions'),
    mpatches.Patch(facecolor=c_feat, edgecolor='white', label='Multi-scale Features'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          facecolor='#2d3436', edgecolor='#666666', labelcolor='white',
          framealpha=0.9)

config_text = (
    "Config: deimv2_hgnetv2_n_pest.yml\n"
    "├─ HGNetv2-B0: return_idx=[1,2,3], use_lab=True\n"
    "├─ HybridEncoder: hidden=128, AIFI@idx[1], exp=0.34, depth=0.5\n"
    "├─ DEIMTransformer: 3 layers, dim=128, ffn=512, points=[4,4,2]\n"
    "└─ Loss: mal=1.0, bbox=5.0, giou=2.0, fgl=0.15, ddf=1.5"
)
ax.text(0.2, 0.2, config_text, ha='left', va='bottom', fontsize=8,
        family='monospace', color='#b2bec3',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#2d3436', edgecolor='#555', alpha=0.8))

plt.tight_layout()
plt.savefig('/root/MYDEIM/deimv2_architecture.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print("Architecture diagram saved to /root/MYDEIM/deimv2_architecture.png")
