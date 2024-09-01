# URL Downloader

URL Downloader is a simple script to download web pages from a list of URLs. It is written in Python and leverages Ray to parallelize the download process.

## Use Cases
You have a large list of web pages to download, but you don't want to download them one by one. This script will download all the web pages in the list for you. 

## Installation
This script requires Python 3.9 or higher. To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
To run the script, run the following command:

```
python download.py --input_file <input_file> --output_dir <output_dir>
```

# Example
To download the web pages in the file `urls.txt` to the directory `output`, run the following command:

```
python3 download.py --url_file ./reddit_urls/RS_2011-01.bz2.deduped.2.txt --output_folder output
```

The input file should be a text file with one URL per line. The output directory will contain the downloaded web pages. The name of each file will be the URL of the web page.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The script borrows heavily from Chip Huyen's [LazyNLP Repo](https://github.com/chiphuyen/lazynlp) 