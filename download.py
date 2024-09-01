import argparse
import glob as glob
import hashlib
import html
import http
import os
import re
import socket
import ssl
import string
import time
import urllib.request
from argparse import ArgumentParser
from collections import Counter

import justext
import ray
import tldextract
from unidecode import unidecode

ray.init()


def parse_html(page):
    """Clean HTML tags for webpages that aren't Gutenberg books"""
    try:
        parts = justext.justext(page, justext.get_stoplist("English"))
    except lxml.etree.ParserError as e:
        print("Page empty")
        return ""
    except UnicodeDecodeError as e:
        print("Can't decode utf-8")
        return ""
    paragraphs = []
    for part in parts:
        if not part.is_boilerplate:
            paragraphs.append(part.text)
    return "\n\n".join(paragraphs)


def clean_html(txt):
    """Clean HTML tags of webpages downloaded
    Use this function for Gutenberg book format.
    """
    style_tag_re = re.compile("<style.*?>[^<>]*?</style>")
    txt = re.sub(style_tag_re, " ", txt)
    script_tag_re = re.compile("<script.*?>[^<>]*?</script>")
    txt = re.sub(script_tag_re, " ", txt)
    doc_tag_re = re.compile("<!DOCTYPE[^<>]*?>")
    txt = re.sub(doc_tag_re, " ", txt)
    html_tag_re = re.compile("<.*?>")
    txt = connect_lines(txt)
    return re.sub(html_tag_re, " ", txt).strip()


def remove_non_alphanumeric(txt):
    """Remove all non-alphanumeric characters, except space, from the text"""
    return re.sub(r"[^a-zA-Z0-9 ]+", "", txt)


def remove_non_alpha(txt):
    """Remove all non-alphabetical characters, except space, from the text"""
    return re.sub(r"[^a-zA-Z ]+", "", txt)


def transliterate(txt):
    """Transliterate foreign characters into its Latin spelling.
    For example, '\u5317\u4EB0' will be transliterated to 'Bei Jing'
    """
    return unidecode(txt)


def collapse_white_spaces(txt):
    """Collapse multiple white spaces into one white space"""
    clean_txt = ""
    prev = None
    for c in txt:
        if c == " " and prev == " ":
            continue
        else:
            clean_txt += c
        prev = c
    return clean_txt


def connect_lines(txt, line_sep="\n"):
    """This happens when you crawl text from a webpage and
    they have random breaking lines mid-sentence.

    This function is to connect those lines.

    Two consecutive lines are separated by line_sep.
    """
    lines = txt.split("\n")

    result, curr = "", ""
    for line in lines:
        line = line.strip()
        if not line:
            if curr:
                result += curr + "\n"
            result += line_sep
            curr = ""
        else:
            curr += line + " "

    return result + curr


def clean_page(page):
    try:
        page = page.decode("utf-8")
    except:
        print("Can't decode")

    page = page.strip()
    if not page:
        return ""
    txt = parse_html(page)
    txt = transliterate(txt)
    txt = html.unescape(txt)
    return txt


def find_unprintable(txt):
    """Find the list of unprintable character
    and return a Counter of them
    """
    printable = set(string.printable)
    unprintable = [c for c in txt if c not in printable]
    return Counter(unprintable)


############################################


def dict_sorted_2_file(dictionary, file, reverse=True):
    with open(file, "w") as out:
        sorted_keys = sorted(dictionary, key=dictionary.get, reverse=reverse)
        for k in sorted_keys:
            out.write("{}\t{}\n".format(k, dictionary[k]))


def get_hash(txt):
    return hashlib.md5(txt.encode()).digest()


def is_initial(token):
    """
    It's an initial is it matches the pattern ([a-z].)*
    """
    return re.match(r"^([a-z]\.)+?$", token.lower()) is not None


def is_positive_number(string, neg=False):
    if not string:
        return False
    if string.isdigit():
        return True
    idx = string.find(".")
    if idx > -1 and idx < len(string) - 1:
        if idx == 0 and neg:
            return False
        new_string = string[:idx] + string[idx + 1 :]
        if new_string.isdigit():
            return True
    rev = string[::-1]
    idx = rev.find(",")

    while idx > 0 and idx % 3 == 0 and rev[:idx].isdigit():
        rev = rev[idx + 1 :]
        idx = rev.find(",")

    if idx == -1 and rev.isdigit():
        return True
    return False


def is_number(string):
    """Return true if:
    integer
    float (both in 32.0323 and .230)
    numbers in the format 239,000,000
    negative number
    """
    if string and string[0] == "-":
        return is_positive_number(string[1:], True)
    return is_positive_number(string)


def get_english_alphabet():
    return set([chr(i) for i in range(ord("a"), ord("z") + 1)])


def sort_files_by_size(files):
    pairs = []
    for file in files:
        size = os.path.getsize(file)
        pairs.append((size, file))
    return sorted(pairs, reverse=True)


def get_filename(path):
    return path[path.rfind("/") + 1 :]


def get_raw_url(url):
    """without http, https, www"""
    idx = url.rfind("//")
    if idx > -1:
        url = url[idx + 2 :]
    if url.startswith("www"):
        url = url[url.find(".") + 1 :]
    return url


def sort_lines(file, reverse=False):
    seen = set()
    with open(file, "r") as f:
        lines = sorted(f.readlines())
    with open(file, "w") as f:
        for line in lines:
            if line not in seen:
                seen.add(line)
                f.write(line)


def to_skip(link, extensions=None, domains=None):
    """domains can be:
    - just the name (as in: google)
    - main domain (as in: google.com)
    - subdomain (as in: news.google.com)
    """
    for ext in extensions:
        if link.endswith(ext):
            return True
    raw_url = get_raw_url(link)
    subdomain, domain, suffix = tldextract.extract(link)
    if domain in domains:
        return True
    if ".".join([domain, suffix]) in domains:
        return True
    if ".".join([subdomain, domain, suffix]) in domains:
        return True
    return False


def download_page(link, context=None, timeout=None, id=None):
    """
    Return code, page
    0: successfully read (write to index)
    1: bad_url (write to bad_url)
    2: unicode error (write to non_ascii_urls)
    3. bad_connection_urls

    When code is not 0, return ''
    """
    try:
        req = urllib.request.Request(link)
    except ValueError as e:
        print(link, "doesn't exist.")
        return 1, ""
    except ConnectionResetError as e:
        print("ConnectionResetError", link)
        return 3, ""

    try:
        if timeout is not None:
            response = urllib.request.urlopen(req, context=context, timeout=timeout)
        else:
            response = urllib.request.urlopen(req, context=context)
    except UnicodeError as e:
        print("UnicodeError for", link)
        return 2, ""
    except urllib.error.HTTPError as e:
        print("Error {} for {}".format(e.code, link))
        return 1, ""
    except urllib.error.URLError as e:
        print("URLError for", link)
        return 1, ""
    except http.client.HTTPException as e:
        print("HTTPException", link)
        return 1, ""
    except http.client.RemoteDisconnected as e:
        print("RemoteDisconnected", link)
        return 1, ""
    except (ConnectionError, socket.timeout) as e:
        print("ConnectionError or Timeout", link)
        return 3, ""

    try:
        page = response.read()
    except http.client.HTTPException as e:
        print("HTTPException", link)
        return 1, ""
    except (ConnectionError, socket.timeout) as e:
        print("ConnectionError or Timeout", link)
        return 3, ""
    return 0, page


def get_current_idx(index_file, links):
    lines = open(index_file, "r").readlines()
    idx = len(lines)
    if idx > 0:
        last_seen = lines[-1].strip()
        while True:
            link = links.readline().strip()
            if link == last_seen:
                break
    return idx, links


@ray.remote
def download_pages(
    link, folder, timeout=30, default_skip=True, extensions=[], domains=[]
):
    """
    link_file (str):
        file contains links to pages to crawl. Each line contains one URL.
    folder (str):
        folder that you want to contain your downloaded pages.
    timeout:
        seconds to wait for a page to respond before abandoning it.

    default_skip (bool):
        True if you want to automatically skip all URLs that contain
        domains and extensions known to be scraper-unfriendly or NSFW.
        See the list of excluded domains at lazynlp/exclude_domains.txt.

        domains can be:
            - just the name (as in: google)
            - main domain (as in: google.com)
            - subdomain (as in: news.google.com)

        See the list of excluded extensions at
        lazynlp/exclude_extensions.txt

        You can also add your own domains and extensions to skip with domains
        and extensions and arguments.

    In the folder:
            Each URL is downloaded into a file, indexed by the order in which
            it is downloaded.
            The first line of each file is the URL.
            The rest is the textual content of the page.

            index.urls contains all the URLs that have been successfully downloaded.
            bad.urls contains the URLs that are bad.
            connection.urls contains the URLs that haven't been downloaded because
                            of connection issues.
            non_ascii.urls contains the URLs that haven't been downloaded because
                            of bad encoding issues.
            empty.urls contains the URLs that have empty textual content.
    """
    idx = 0

    os.makedirs(folder, exist_ok=True)

    index = open(os.path.join(folder, "index.urls"), "a")
    skipped_urls = open(os.path.join(folder, "skip.urls"), "a")
    bad_connection_urls = open(os.path.join(folder, "connection.urls"), "a")
    bad_urls = open(os.path.join(folder, "bad.urls"), "a")
    non_ascii_urls = open(os.path.join(folder, "non_ascii.urls"), "a")
    empty_urls = open(os.path.join(folder, "empty.urls"), "a")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    hashed = hashlib.sha1()

    if default_skip:
        if args.extensions is None:
            ext_lines = open(f"./url_list/exclude_extensions.txt", "r").readlines()
            
        else:
            ext_lines = open(args.extensions, "r").readlines()
            
        if args.domains is None:
            domain_lines = open(f"./url_list/exclude_domains.txt", "r").readlines()
        else:
            domain_lines = open(args.domains, "r").readlines()
        
        extensions.extend([line.strip() for line in ext_lines])
        domains.extend([line.strip() for line in domain_lines])
        

    link = link.strip()
    if to_skip(link, extensions, domains):
        skipped_urls.write(link + "\n")
        print("Skip", link)

    code, page = download_page(link, ctx, timeout)
    if code == 1:
        bad_urls.write(link + "\n")
    elif code == 2:
        non_ascii_urls.write(link + "\n")
    elif code == 3:
        bad_connection_urls.write(link + "\n")
    if code > 0:
        print("Bad page", link)

    txt = clean_page(page)

    if not txt:
        print("Empty page", link)
        empty_urls.write(link + "\n")

    print(idx, link)
    hashed.update(str(time.time()).encode())
    name = hashed.hexdigest()
    with open(f"{folder}/{idx}_{name}.txt", "w") as out:
        out.write(link + "\n" + txt)

    print(find_unprintable(txt))
    index.write("{}\n".format(link))
    idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url_file",
        type=str,
        help="file contains links to pages to crawl. Each line contains one URL.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="folder that you want to contain your downloaded pages.",
    )
    parser.add_argument(
        "--exclude_domains", type=str, help="file contains domains to skip"
    )
    parser.add_argument(
        "--exclude_extensions", type=str, help="file contains extensions to skip"
    )
    args = parser.parse_args()
    url_file = args.url_file
    output_folder = args.output_folder

    with open(url_file, "r") as f:
        links = f.readlines()

    output = [
        download_pages.remote(
            link,
            output_folder,
            timeout=30,
            default_skip=True,
            extensions=[],
            domains=[],
        )
        for link in links
    ]
    output = ray.get(output)
    print(output)
