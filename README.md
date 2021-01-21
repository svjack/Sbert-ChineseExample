<!--
# Sbert-ChineseExample
Sentence-Transformers Information Retrieval example on Chinese

# Weighted-W2V-Senti
Douban Movie Short Comments Dataset Sentiment Analysis By Sentiment Weighted Word2Vec
-->
<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!--
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->
  
  <!--
  <h3 align="center">Best-README-Template</h3>
  <h3 align="center">Weighted-W2V-Senti</h3>
  -->
  <h3 align="center">Sbert-ChineseExample</h3>

  <p align="center">
    <!--An awesome README template to jumpstart your projects!-->
    <!--Time Series Forecast use Scikit-Hts with params fine-tuned by Hyperopt on multi-feature.-->
    <!--Douban Movie Short Comments Dataset Sentiment Analysis By Sentiment Weighted Word2Vec-->
    Sentence-Transformers Information Retrieval example on Chinese
    <br />
    <!--
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    -->
    <br />
    <br />
    <!--
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    <a href="notebook/word2vec-sent-cls.ipynb">View Demo</a>
    -->
    <!--
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
    -->
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  <!--
  * [Prerequisites](#prerequisites)
  -->
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->


<!--
There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need.
-->
<!--
Time Series Forecast is useful in retail analysis in data analysis and mining. When it comes to hts data --- a data format that apply decoposition on massive
data, Many models have built on them. You can have a look at https://otexts.com/fpp2/hts.html
Nowadays, Many R packages have be developed. And you can find Scikit-Hts this python framework to perform similar tasks.
But there are many limits as follows:
* R packages such as gts and Scikit-Hts seems only support uni-feature time series, that you can not add other feature related with time series.
* Scikit-Hts have some Bugs with bottom up situation, which can be seen at https://github.com/carlomazzaferro/scikit-hts/issues/35
* Complex params fine-tuned on Prophet and the above framework (Scikit-Hts) may be difficult.

Sentiment Analysis is a basic task in NLP. There are some common used feature construction method for this task.
Such as : Tf-idf Word2Vec and some basic method based on sentiment dictionary on words.
Douban Movie Short Comments Dataset is a classical dataset for Sentiment Analysis.
You can take a look at https://www.kaggle.com/weiyunchen/nlp123, which use Tf-idf as feature and build DNN on Douban Movie Short Comments Dataset.

This work is inspired by that jupyter notebook. Also use a sample of the full datasets.
The Difference is that rather than use Tf-idf or average Word2Vec vectors as sentence vector, 
This project use Word2Vec and try to merge sentiment dictionary with Word2Vec.

The idea is simple, translation the Word2Vec vector to positive space and weighted vector by the sentiment dictionary.
The key is the construction of sentiment dictionary. It use simple voting on the full dataset. And filter out some invalid 
words (contains stopwords and others --- may be gap between sample and population)

The conclusion show that the Sentiment Weighted Word2Vec feature improve 6% balance accuracy than simple average Word2Vec feature
in Random Forest Model.
-->
Sentence Transformers is a multilingual sentence embedding generate framework, which provides an easy method to compute dense
vector representations for sentences and paragraphs (based on HuggingFace Transformers)

This repository target at ms_macro like task on a Chinese dataset, train bi_encoder and cross_encoder, with the help of
elasticsearch easy interface on pandas to build serlizable conclusion.

<!--
Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue.

A list of commonly used resources that I find helpful are listed in the acknowledgements.
-->
### Built With
<!--
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com)

* [Prophet](https://www.prophet.com/)
* [Scikit-Hts](https://github.com/carlomazzaferro/scikit-hts)
* [Hyperopt](https://github.com/hyperopt/hyperopt)

* [Gensim](https://github.com/RaRe-Technologies/gensim)
* [Wikipedia2Vec](https://github.com/wikipedia2vec/wikipedia2vec)
-->
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
* [Elasticsearch](https://github.com/elastic/elasticsearch)
* [Faiss](https://github.com/facebookresearch/faiss)

<!-- GETTING STARTED -->
## Getting Started
<!--
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->

<!--
### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```
-->

### Installation
* pip
```sh
pip install -r requirements.txt
```
* install Elasticsearch and start service
<!--
1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
3. Install NPM packages
```sh
npm install
```
4. Enter your API in `config.js`
```JS
const API_KEY = 'ENTER YOUR API';
```
-->


<!-- USAGE EXAMPLES -->
## Usage
<!--
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
-->

<p>
<a href="https://github.com/brightmart/nlp_chinese_corpus#4%E7%A4%BE%E5%8C%BA%E9%97%AE%E7%AD%94json%E7%89%88webtext2019zh-%E5%A4%A7%E8%A7%84%E6%A8%A1%E9%AB%98%E8%B4%A8%E9%87%8F%E6%95%B0%E6%8D%AE%E9%9B%86">
1. Download Data from google drive</a>
</p>

<p>
<a href="script/bi_encoder/bi-encoder-data.py">2. bi_encoder data prepare</a>
</p>

<p>
<a href="script/bi_encoder/bi-encoder-batch.py">3. train bi_encoder</a>
</p>

<p>
<a href="script/cross_encoder/try_sbert_neg_sampler.py">4. cross_encoder train data prepare</a>
</p>

<p>
<a href="script/cross_encoder/try_sbert_neg_sampler_valid.py">5. cross_encoder valid data prepare</a>
</p>

<p>
<a href="script/cross_encoder/cross_encoder_random_on_multi_eval.py">6. train cross_encoder</a>
</p>

<p>
<a href="script/cross_encoder/valid_cross_encoder_on_bi_encoder.py">7. show bi_encoder cross_encoder inference</a>
</p>

<!-- ROADMAP -->
## Roadmap
<!--
See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).
-->
<!--
<p>
<a href="notebooks/time-series-explore.ipynb">Time Series Data Analysis Notebook</a>
</p>
You can also step by step run Jupyter Notebooks in /notebooks dir.
-->
1 This repository use edited es-pandas interface (support vector serlized) to have a simple manipulate on elasticsearch by pandas.
<br />
2 try_sbert_neg_sampler.py sample hard negative samples drived from class provide by
https://guzpenha.github.io/transformer_rankers/
can also use elastic search to generate hard samples , relate functions have defined in valid_cross_encoder_on_bi_encoder.py
<br />
3 Before training your dataset on cross_encoder, should take a look at the semantic similarity between different questions.
Combine some samples with similar semantic may give help.
<br />
4 Add some toolkit to Sbert to support multi-class-evaluation (as dictionary)
 <!-- CONTRIBUTING -->
## Contributing
<!--
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
-->


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link: [https://github.com/svjack/Sbert-ChineseExample](https://github.com/svjack/Sbert-ChineseExample)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
* [Elasticsearch](https://github.com/elastic/elasticsearch)
* [Faiss](https://github.com/facebookresearch/faiss)
* [Transformer Rankers](https://github.com/Guzpenha/transformer_rankers)
* [es_pandas](https://github.com/fuyb1992/es_pandas)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png

