## Motif sampling (repository for paper)

<br/> This repository contains the scripts that generate the main figures reported in the paper: <br/>


Facundo Memoli, Hanbaek Lyu, and David Sivakoff,\
[*"Sampling random graph homomorphisms and applications to network data analysis*"](https://arxiv.org/abs/1910.09483) (arXiv 2019)


&nbsp;

For a more user-friendly repository, please see [NNetwork package repository](https://github.com/HanbaekLyu/NNetwork).\
Some of our code is also available as the python package [**NNetwork**](https://pypi.org/project/NNetwork/) on pypi.
 

&nbsp;

![](Figures/fig1.png)
&nbsp;
![](Figures/fig2.png)
&nbsp;
![](Figures/fig3.png)
&nbsp;
![](Figures/fig4.png)
&nbsp;
![](Figures/fig5.png)
&nbsp;
![](Figures/fig6.png)
&nbsp;

## Usage

First add network files for UCLA, Caltech, MIT, Harvard to Data/Networks_all_NDL\
Ref: Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter,\
*Comparing community structure tocharacteristics in online collegiate social networks.* SIAM Review, 53:526–543, 2011.
&nbsp;

Then copy & paste the ipynb notebook files into the main folder. Run each Jupyter notebook and see the instructions therein. 

## File description 

  1. **src.dyn_emb.py** : main source file for MCMC motif sampling and computing MACC and conditional homomorphism density profiles. 
  2. **src.dyn_emb_app.py**: application script of main algorithms
  3. **src.dyn_emb_facebook.py**: application script for Facebook100 dataset 
  4. **src.WAN_classifier.py**: application script for Word Adjacency Networks dataset 
  5. **src.helper_functions.py**: helper functions for plotting and auxiliary computation 
  6. **motif_sampling_ex.ipynb**: Jupyter notebook for motif sampling plots
  7. **subgraph_classification.ipynb**: Jupyter notebook for subgraph classification experiments
 
## Authors

* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

