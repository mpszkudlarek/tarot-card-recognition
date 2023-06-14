from trainFunctions import *
from transforms import *
from NetworkModel import *
from MakePrediction import *


def main():
    device = print_info()
    NUM_EPOCHS = 10
    # wczytujemy dane
    dataSetDir = "DataSet/"
    train_data_trivial, class_names, class_dict, test_data_Trivial = load_dataSet(data_trnasform_ColorJitter,
                                                                                  dataSetDir)  # jako agrument podajemy transformację z pliku transforms.py oraz ścieżkę do datasetu
    train_dataloader_trivial, test_dataloader_trivial = getDataLoader(train_data_trivial, test_data_Trivial)
    torch.manual_seed(42)

    model_0 = TinyVGG(input_shape=3,  # number of color channels (3 for RGB) 
                      hidden_units=10,
                      output_shape=len(class_names)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
    # poniższy kod służy do wczytywania sieci w celu jej dalszego uczenia
    # checkpoint = torch.load("model_trivial_0.pt")
    # model_0.load_state_dict(checkpoint['model_state_dict']) 
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model_0.train()

    # rozpoczynamy uczenie
    model_0_results = train(device,
                            'trivial',  # nazwa transformacji, która będzie potem występować w nazwie pliku, w którym zapisana jest siećl.
                            model=model_0,
                            train_dataloader=train_dataloader_trivial,
                            test_dataloader=test_dataloader_trivial,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # poniższy kod służy do wczytania sieci i zrobienia predykcji
    # model_0.load_state_dict(torch.load("model_trivial_0.pt")['model_state_dict']) 
    # model_0.eval()
    # pred_and_plot_image(device, model_0, "tarot_card_0.jpg", class_names, custom_image_tr)
    # pred_and_plot_image(device, model_0, "tarot_card_1.jpg", class_names, custom_image_tr)
    # pred_and_plot_image(device, model_0, "tarot_card_2.jpg", class_names, custom_image_tr)

    # a ta część robi to samo, tylko dla wszystkich zdjęć, które znajdują sie w danym folderze
    # for fileName in os.listdir('detected_cards/'):
    #     pred_and_plot_image(device, model_0, 'detected_cards/' + fileName, class_names, custom_image_tr)


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
