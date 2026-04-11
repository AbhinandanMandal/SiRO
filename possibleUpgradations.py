
"""
Possible Upgradations in PiRO code
-----------------------------------------------------
In the architecture part
replacing vgg with resnet50, efficientnet, vit
adding projection head (simCLR style)
adding positional encoding
instead of custom multiheadattention using transformer
for better aggregation, instead of mean we can use attention pooling, CLS token
mean pooling line: MVOBJEmbs = F.normalize(torch.mean(objContext, 1), dim=1, p=2)
we can also add nt-xent loss


upgrading for vgg_pan_dualembd.py
cross attention : let category guide object
objContext = CrossAttention(objEmbs, clsEmbs)
replace mean pooling
using ViT



In loss function
probably need to upgrade and think of some new physics based loss function
maybe we can also add some PINN stuffs into it to make it more efficient and effective
need to think of a better approach than current existing ones.


"""