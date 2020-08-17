from dataloader.cogsci import CogSciDataReader


def main():
    path = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data"
    datareader = CogSciDataReader(path)
    for item in datareader:
        print(item.text.strip())


if __name__ == '__main__':
    main()
