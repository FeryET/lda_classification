from dataloader import CogSciData, DataReader
def main():
    path = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data"
    datareader = DataReader(path, data_type=CogSciData)
    print(datareader.to_pandas())


if __name__ == '__main__':
    main()
