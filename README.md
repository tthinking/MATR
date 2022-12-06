# MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer (IEEE TIP 2022).

This is the official implementation of the MATR model proposed in the paper ([MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer](https://ieeexplore.ieee.org/document/9844446)) with Pytorch.

<h1 dir="auto"><a id="user-content-requirements" class="anchor" aria-hidden="true" href="#requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Requirements</h1>
<ul dir="auto">
<li>CUDA 11.4</li>
<li>conda 4.10.1</li>
<li>Python 3.8.12</li>
<li>PyTorch 1.9.1</li>
<li>timm 0.4.12</li>
<li>tqdm</li>
<li>glob</li>
<li>pandas</li>
</ul>

# Tips:
<strong>Dealing with RGB input:</strong>
Refer to [DPCN-Fusion](https://github.com/tthinking/DPCN-Fusion/blob/master/test.py).

<strong>Dataset is </strong> [here](http://www.med.harvard.edu/AANLIB/home.html).

The code for <strong>evaluation metrics</strong> is [here](https://github.com/tthinking/MATR/tree/main/evaluation).


# Cite the paper
If this work is helpful to you, please cite it as:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@ARTICLE{Tang_2022_MATR,
  author={Tang, Wei and He, Fazhi and Liu, Yu and Duan, Yansong},
  journal={IEEE Transactions on Image Processing}, 
  title={MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer}, 
  year={2022},
  volume={31},
  number={},
  pages={5134-5149},
  doi={10.1109/TIP.2022.3193288}}"><pre class="notranslate"><code>@ARTICLE{Tang_2022_MATR,
  author={Tang, Wei and He, Fazhi and Liu, Yu and Duan, Yansong},
  journal={IEEE Transactions on Image Processing}, 
  title={MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer}, 
  year={2022},
  volume={31},
  number={},
  pages={5134-5149},
  doi={10.1109/TIP.2022.3193288}}
</code></pre></div>

If you have any questions,  feel free to contact me (<a href="mailto:weitang2021@whu.edu.cn">weitang2021@whu.edu.cn</a>).
