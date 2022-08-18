import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from caffe2.python.core import Net
import yfinance as yf
import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

from GUI import *

import matplotlib.pyplot as plt
import numpy as np

class Tela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #self.object.signal.conect(self.slot)
        self.ui.btn_validarAcao.clicked.connect(self.validarAcao)
        self.ui.btn_confirmarDataInicialFinal.clicked.connect(self.confirmarDataInicialFinal)
        self.ui.btn_iniciarTreinamento.clicked.connect(self.treinoDaIA)

    @QtCore.pyqtSlot()
    def validarAcao(self):
        nomeDaAcao = self.ui.comboBox_acao.currentText()
        if nomeDaAcao == "LAME4.SA":
            self.ui.lbl_resultadoNomeEmpresa.setText("Lojas Americanas S.A.")
            self.ui.lbl_resultadoSetorEmpresa.setText("Consumer Cyclical")
            self.ui.lbl_resultadoPaisEmpresa.setText("Brazil")
        elif nomeDaAcao == "MGLU3.SA":
            self.ui.lbl_resultadoNomeEmpresa.setText("Magazine Luiza S.A.")
            self.ui.lbl_resultadoSetorEmpresa.setText("Consumer Cyclical")
            self.ui.lbl_resultadoPaisEmpresa.setText("Brazil")
        elif nomeDaAcao == "JBSAY":
            self.ui.lbl_resultadoNomeEmpresa.setText("JBS S.A.")
            self.ui.lbl_resultadoSetorEmpresa.setText("Consumer Defensive")
            self.ui.lbl_resultadoPaisEmpresa.setText("Brazil")
        else:
            self.ui.lbl_resultadoNomeEmpresa.setText("Amazon.com, Inc.")
            self.ui.lbl_resultadoSetorEmpresa.setText("Consumer Cyclical")
            self.ui.lbl_resultadoPaisEmpresa.setText("United States")

        print(self.ui.spinBox_escolherNeuroniosPorCamada.value())

        return nomeDaAcao



    @QtCore.pyqtSlot()
    def confirmarDataInicialFinal(self):
        dataInicial = self.ui.dateEdit_dataInicial.date()
        dataFinal = self.ui.dateEdit_dataFinal.date()
        nomeDaAcao = self.validarAcao()
        print(nomeDaAcao)

        dataInicial = str(dataInicial)
        dataInicial = dataInicial.replace("PyQt5.QtCore.QDate", "")
        dataInicial = dataInicial.replace("(", "")
        dataInicial = dataInicial.replace(")", "")
        dataInicial = dataInicial.replace(", ", "-")
        #print(dataInicial)

        dataFinal = str(dataFinal)
        dataFinal = dataFinal.replace("PyQt5.QtCore.QDate", "")
        dataFinal = dataFinal.replace("(", "")
        dataFinal = dataFinal.replace(")", "")
        dataFinal = dataFinal.replace(", ", "-")
        # print(dataFinal)

        oibr = yf.Ticker(f'{nomeDaAcao}')
        data = yf.download(f'{nomeDaAcao}', start=dataInicial, end=dataFinal)
        data = data.Close

        # Plotar apenas teste
        plt.figure(figsize=(18, 6))
        plt.plot(data, '-')
        plt.xlabel('ANOS')
        plt.ylabel('VALOR R$')
        plt.title(f'{nomeDaAcao}')
        plt.show()


    def treinoDaIA(self):
        nomeDaAcao = self.validarAcao()

        # Exibir informações das ações do Itaú
        oibr = yf.Ticker(f'{nomeDaAcao}')
        # petr.info

        # Coletar dados da Oi
        data = yf.download(f'{nomeDaAcao}', start='2016-01-01', end='2020-11-19')
        # data

        # Coletar somente o fechamento diário
        data = data.Close

        jan = 50

        data_final = np.zeros([data.size - jan, jan + 1])

        for i in range(len(data_final)):
            for j in range(jan + 1):
                data_final[i][j] = data.iloc[i + j]

        # Normalizar entre 0 e 1
        max = data_final.max()
        min = data_final.min()
        dif = data_final.max() - data_final.min()
        data_final = (data_final - data_final.min()) / dif

        x = data_final[:, :-1]
        y = data_final[:, -1]

        # Entrada do treinamento
        training_input = torch.FloatTensor(x[:850, :])
        # Saída do treinamento
        training_output = torch.FloatTensor(y[:850])

        # Entrada do teste
        test_input = torch.FloatTensor(x[850:, :])
        # Saída do teste
        test_output = torch.FloatTensor(y[850:])

        # print(test_input)
        # print(test_output)

        # Classe do modelo da Rede Neural
        class Net(torch.nn.Module):
            def __init__(self, input_size, hidden_size):
                super(Net, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(self.hidden_size, 1)

            def forward(self, x):
                hidden = self.fc1(x)
                relu = self.relu(hidden)
                output = self.fc2(relu)
                output = self.relu(output)
                return output

        # Criar a instância do modelo
        input_size = training_input.size()[1]
        hidden_size = self.ui.spinBox_escolherNeuroniosPorCamada.value()
        model = Net(input_size, hidden_size)
        print(f'Entrada: {input_size}')
        print(f'Escondida: {hidden_size}')
        print(model)

        # Critério de erro
        criterion = torch.nn.MSELoss()

        # Criando os paramêtros (learning rate[obrigatória] e momentum[opcional])
        lr = self.ui.doubleSpinBox_escolhaDeTaxaDeAprendizagem.value()
        momentum = self.ui.doubleSpinBox_taxaDeMomentun.value()
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum)

        # Para visualizar os pesos
        for param in model.parameters():
            # print(param)
            pass

        # Treinamento
        model.train()
        epochs = self.ui.spinBox_escolherEpocas.value()
        errors = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Fazer o forward
            y_pred = model(training_input)
            # Cálculo do erro
            loss = criterion(y_pred.squeeze(), training_output)
            errors.append(loss.item())
            if epoch % 1000 == 0:
                print(f'Epoch: {epoch}. Train loss: {loss.item()}.')
            # Backpropagation
            loss.backward()
            optimizer.step()

        # Testar o modelo já treinado
        model.eval()
        y_pred = model(test_input)
        after_train = criterion(y_pred.squeeze(), test_output)
        print('Test loss after Training', after_train.item())

        # Gráficos de erro e de previsão
        def plotcharts(errors):
            errors = np.array(errors)
            lasterrors = np.array(errors[-25000:])
            plt.figure(figsize=(18, 6))
            graf01 = plt.subplot(1, 2, 1)  # nrows, ncols, index
            graf01.set_title('Errors')
            plt.plot(errors, '-')
            plt.xlabel('Epochs')
            graf03 = plt.subplot(1, 2, 2)
            graf03.set_title('Predicted')
            a = plt.plot(test_output.numpy(), 'y-', label='Real')
            a = plt.plot(y_pred.detach().numpy(), 'b-', label='Predicted')
            plt.legend(loc=2)
            plt.show()

        plotcharts(errors)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mostraNaTela = Tela()
    mostraNaTela.show()
    sys.exit(app.exec_())