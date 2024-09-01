import argparse
import hashlib
import html
import os
import re
import ssl
import time
import urllib.request
from collections import Counter

import justext
import ray
import tldextract
from unidecode import unidecode

class WebCrawler:
    def __init__(self, output_folder, timeout=30, default_skip=True, extensions=None, domains=None):
        self.output_folder = output_folder
        self.timeout = timeout
        self.default_skip = default_skip
        self.extensions = extensions or []
        self.domains = domains or []
        self.setup_output_folder()
        self.ssl_context = self.setup_ssl_context()

    def setup_output_folder(self):
        os.makedirs(self.output_folder, exist_ok=True)
        self.index_file = open(os.path.join(self.output_folder, "index.urls"), "a")
        self.skipped_urls_file = open(os.path.join(self.output_folder, "skip.urls"), "a")
        self.bad_connection_urls_file = open(os.path.join(self.output_folder, "connection.urls"), "a")
        self.bad_urls_file = open(os.path.join(self.output_folder, "bad.urls"), "a")
        self.non_ascii_urls_file = open(os.path.join(self.output_folder, "non_ascii.urls"), "a")
        self.empty_urls_file = open(os.path.join(self.output_folder, "empty.urls"), "a")

    def setup_ssl_context(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def load_exclusion_lists(self):
        if self.default_skip:
            with open("./url_list/exclude_extensions.txt", "r") as f:
                self.extensions.extend([line.strip() for line in f])
            with open("./url_list/exclude_domains.txt", "r") as f:
                self.domains.extend([line.strip() for line in f])

    def to_skip(self, link):
        for ext in self.extensions:
            if link.endswith(ext):
                return True
        subdomain, domain, suffix = tldextract.extract(link)
        if domain in self.domains or f"{domain}.{suffix}" in self.domains or f"{subdomain}.{domain}.{suffix}" in self.domains:
            return True
        return False

    def download_page(self, link):
        try:
            req = urllib.request.Request(link)
            response = urllib.request.urlopen(req, context=self.ssl_context, timeout=self.timeout)
            return 0, response.read()
        except Exception as e:
            print(f"Error downloading {link}: {str(e)}")
            return 1, ""

    def clean_page(self, page):
        try:
            page = page.decode("utf-8")
        except:
            print("Can't decode")
            return ""

        parts = justext.justext(page, justext.get_stoplist("English"))
        paragraphs = [part.text for part in parts if not part.is_boilerplate]
        txt = "\n\n".join(paragraphs)
        txt = unidecode(txt)
        txt = html.unescape(txt)
        return txt

    def process_link(self, link, idx):
        link = link.strip()
        if self.to_skip(link):
            self.skipped_urls_file.write(f"{link}\n")
            print(f"Skip {link}")
            return

        code, page = self.download_page(link)
        if code != 0:
            self.bad_urls_file.write(f"{link}\n")
            print(f"Bad page {link}")
            return

        txt = self.clean_page(page)

        if not txt:
            print(f"Empty page {link}")
            self.empty_urls_file.write(f"{link}\n")
            return

        print(f"{idx} {link}")
        hashed = hashlib.sha1(str(time.time()).encode()).hexdigest()
        with open(f"{self.output_folder}/{idx}_{hashed}.txt", "w") as out:
            out.write(f"{link}\n{txt}")

        self.index_file.write(f"{link}\n")

@ray.remote
def process_link_wrapper(crawler, link, idx):
    return crawler.process_link(link, idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_file", type=str, required=True, help="File containing URLs to crawl")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store downloaded pages")
    parser.add_argument("--exclude_domains", type=str, help="File containing domains to skip")
    parser.add_argument("--exclude_extensions", type=str, help="File containing extensions to skip")
    args = parser.parse_args()

    ray.init()

    crawler = WebCrawler(args.output_folder)
    crawler.load_exclusion_lists()

    with open(args.url_file, "r") as f:
        links = f.readlines()

    tasks = [process_link_wrapper.remote(crawler, link, idx) for idx, link in enumerate(links)]
    ray.get(tasks)

if __name__ == "__main__":
    main()