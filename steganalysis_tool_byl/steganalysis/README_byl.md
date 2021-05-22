## Auto is the default algorithm developed in 2021, which I think is the more convenient one to use:
python aletheia.py auto DATA_PATH

example: python aletheia.py auto sample_images/alaska2

## HiddenNet use ATS, if you want to use it, please run:
python aletheia.py ats SELECTED_ALGORIGHTM HYPERAMETER( such as 0.4 ) srm DATA_PATH

example: python aletheia.py ats hugo-sim 0.4 srm sample_images/alaska2jpg


Note: for those examples, I have tested them, please be careful about how to use them. 
you will get results like this:

                       LSBR      LSBM  SteganoGAN  HILL *
---------------------------------------------------------
00839_hill.png          0.0      [0.8]     0.0     [1.0]
04686.png               0.0       0.0      0.0      0.0
25422.png               0.0       0.0      0.0      0.0
27693_steganogan.png   [0.9]     [1.0]    [1.0]    [0.9]
34962_hill.png          0.0       0.0      0.0     [0.5]
36466_steganogan.png   [0.9]     [1.0]    [1.0]    [1.0]
37831_lsbm.png         [1.0]     [1.0]     0.0     [0.7]
55453_lsbm.png         [0.6]     [0.9]     0.0     [0.9]
67104_steganogan.png   [0.9]     [0.9]    [1.0]    [0.8]
74051_hill.png          0.0       0.0      0.0     [0.9]
74648_lsbm.png         [1.0]     [1.0]     0.0     [0.6]
74664.png               0.0       0.0      0.0      0.0


In the end, please don't forget to cite the original work from the author

```
@software{daniel_lerch_hostalot_2021_4655945,
  author       = {Daniel Lerch-Hostalot},
  title        = {Aletheia},
  month        = apr,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.4655945},
  url          = {https://doi.org/10.5281/zenodo.4655945}
}
```

Also, other related works that we could refer to
References
[1]. Attacks on Steganographic Systems. A. Westfeld and A. Pfitzmann. Lecture Notes in Computer Science, vol.1768, Springer-Verlag, Berlin, 2000, pp. 61−75.
[2]. Reliable Detection of LSB Steganography in Color and Grayscale Images. Jessica Fridrich, Miroslav Goljan and Rui Du. Proc. of the ACM Workshop on Multimedia and Security, Ottawa, Canada, October 5, 2001, pp. 27-30.
[3]. Detection of LSB steganography via sample pair analysis. S. Dumitrescu, X. Wu and Z. Wang. IEEE Transactions on Signal Processing, 51 (7), 1995-2007.
[4]. Unsupervised Steganalysis Based on Artificial Training Sets. Daniel Lerch-Hostalot and David Megías. Engineering Applications of Artificial Intelligence, Volume 50, 2016, Pages 45-59.
[5]. On steganalysis of random LSB embedding in continuous-tone images. Sorina Dumitrescu, Xiaolin Wu and Nasir D. Memon. Proceedings of ICIP, 2002, Rochester, NY, pp.641-644.
[6]. Revisiting Weighted Stego-Image Steganalysis. Andrew D. Ker and Rainer Böhme. Security, Forensics, Steganography, and Watermarking of Multimedia Contents X, Proc. SPIE Electronic Imaging, vol. 6819, San Jose, CA, pp. 0501-0517, 2008.
[7]. A general framework for the structural steganalysis of LSB replacement. Andrew D. Ker. Proceedings 7th Information Hiding Workshop, LNCS vol. 3727, pp. 296-311, Barcelona, Spain. Springer-Verlag, 2005.
[8]. Adaptive steganalysis of Least Significant Bit replacement in grayscale natural images. L. Fillatre. Signal Processing, IEEE Transactions on, vol. 60, issue 2, pp. 556-569, February 2012.
