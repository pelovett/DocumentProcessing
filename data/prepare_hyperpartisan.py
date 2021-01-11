import sys
import os
import xml.etree.ElementTree as ET


def main(file_path):
    assert os.path.isdir(file_path)
    if file_path[-1] != '/':
        file_path += '/'
    needed = ['training', 'validation']
    for directory in needed:
        if not os.path.isdir(file_path + directory):
            os.makedirs(file_path + directory)

    for subset in ['training', 'validation']:
        tree = ET.parse(file_path+'articles-'+subset +
                        '-bypublisher-20181122.xml')

        output_name = file_path+subset+'/X.txt'
        with open(output_name, 'w') as out_file:
            i = 0
            for article in tree.getroot():
                article_id = article.attrib['id']
                article_text = article.attrib['title'] + ' '
                for child in article:
                    article_text += ''.join(child.itertext()) + ' '
                article_text = article_text.replace('\n', ' ')
                article_text += '\n'
                out_file.write(article_text)
                i += 1
        print(f'Processed {i} articles')

        tree = ET.parse(file_path + 'ground-truth-' +
                        subset + '-bypublisher-20181122.xml')
        with open(file_path+subset+'/Y.txt', 'w') as out_file:
            for article in tree.getroot():
                article_id = article.attrib['id']
                article_hype = article.attrib['hyperpartisan']
                article_bias = article.attrib['bias']
                out_file.write('\t'.join(
                    [article_id, article_hype, article_bias]) + '\n')
        print(f'Completed subset: {subset}')


if __name__ == "__main__":
    main(sys.argv[1])
