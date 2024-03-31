<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-federated-didactic/blob/master/assets/logo.jpg" />
</h1>

<h3 align="center">
  Educational Example of Federated Recommendation System
</h3>

<p align="center">Exemplo de medidas de justiça do usuário em Sistemas de Recomendação </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-federated-didactic?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-federated-didactic/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-federated-didactic?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

> É proposto um script em Python para simular o funcionamento de um sistema de recomendação federado.

O objetivo deste projeto é demonstrar um exemplo didático de um sistema de recomendação federado utilizando aprendizado de máquina. A abordagem federada permite treinar modelos de machine learning de forma descentralizada, utilizando dados distribuídos entre vários clientes sem a necessidade de centralizar esses dados em um único servidor. Esse método aumenta a privacidade e segurança dos dados, uma vez que as informações sensíveis dos usuários não precisam ser compartilhadas ou transferidas para um local central. Neste exemplo, são implementadas funções para treinar modelos locais em diferentes clientes, agregando posteriormente as atualizações dos modelos locais ao modelo global sem compartilhar os dados diretamente, mas sim através da agregação de parâmetros ou gradientes dos modelos. O sistema busca oferecer recomendações personalizadas, mantendo ao mesmo tempo a privacidade dos dados dos usuários.

### Funções Principais do Projeto

| Função                             | Descrição                                                                                                                                                                        |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `print_matriz_com_precisao`        | Imprime matrizes de avaliações ou recomendações com precisão de duas casas decimais.                                                                                             |
| `visualize_model`                  | Gera uma visualização da arquitetura da rede neural, mostrando neurônios e conexões para facilitar a compreensão estrutural.                                                     |
| `treinar_modelo_global`            | Treina o modelo global com avaliações iniciais para aproximar saídas esperadas, preparando-o para a distribuição aos clientes.                                                   |
| `treinar_modelos_locais`           | Simula aprendizado local treinando modelos em dados específicos de clientes e gerando novas avaliações.                                                                         |
| `agregar_modelos_locais_ao_global_pesos` | Agrega atualizações dos modelos locais ao global pela média dos parâmetros, promovendo melhorias baseadas em aprendizados locais.                                               |
| `agregar_modelos_locais_ao_global_gradientes` | Atualiza o modelo global pela média dos gradientes dos modelos locais, permitindo ajustes finos baseados em tendências de aprendizado locais.                                   |
| `mostrar_calculos_agregacao_modelos` | Exibe os cálculos de agregação dos modelos locais ao global, oferecendo uma visão detalhada dos parâmetros individuais e da média calculada para fins educativos e de verificação. |



### Etapas do Script Principal (main)

1. **Geração da Matriz de Avaliações Inicial**: Criação de uma matriz que simula as avaliações dos usuários a diversos itens.
2. **Treinamento do Modelo Global no Servidor**: O modelo global aprende a prever avaliações a partir da matriz inicial.
3. **Distribuição do Modelo Global aos Clientes Locais**: O modelo é enviado para clientes, onde cada um tem dados únicos de avaliação.
4. **Treinamento Local nos Clientes**: Cada cliente treina o modelo com seus dados para refinar as previsões.
5. **Geração de Novas Avaliações Locais**: Os clientes geram avaliações para itens não previamente avaliados.
6. **Agregação dos Modelos Locais ao Modelo Global**: Os aprendizados locais são agregados ao modelo global, melhorando suas previsões.
7. **Avaliação do Desempenho do Modelo**: Mensuração do progresso do modelo global por meio da métrica MSE antes e após a agregação.
8. **Visualização dos Modelos**: Visualizações gráficas ajudam a entender as estruturas do modelo ao longo do processo.


## :rocket: Instalação e Execução

Para instalar e executar este projeto, siga estes passos:

1. **Clone o Repositório**
   ```
   git clone https://github.com/ravarmes/recsys-federated-didactic.git
   ```

2. **Instale as Dependências**
   Dentro do diretório do projeto, execute:
   ```
   pip install -r requirements.txt
   ```

3. **Execute o Projeto**
   ```
   python recsys-federated-didatic.py
   ```

## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: