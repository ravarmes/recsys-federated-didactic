import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.activation(self.layer2(x)) * 4 + 1  # Scale sigmoid output to range [1, 5]
        return x
    
def print_matriz_com_precisao(matriz):
    matriz_np = matriz.detach().numpy() if isinstance(matriz, torch.Tensor) else np.array(matriz)
    for linha in matriz_np:
        print("[", " ".join(f"{valor:.2f}" for valor in linha), "]")

def visualize_model(model, title='Model'):
    plt.figure(figsize=(8, 6))  # Define o tamanho da figura

    G = nx.DiGraph()
    total_input_nodes = 5
    total_hidden_nodes = 2
    total_output_nodes = 5

    # Mantém os neurônios da camada de entrada e saída como estão
    node_pos = {
        **{f'Input Layer_{i}': (0, -i) for i in range(total_input_nodes)},
        **{f'Output Layer_{i}': (2, -i) for i in range(total_output_nodes)}
    }

    # Centraliza e aumenta o espaçamento entre os neurônios da camada oculta
    # Calcula o offset para centralizar considerando o novo espaçamento
    offset_hidden = -(total_input_nodes - 2) / 2  # Ajusta o início para os neurônios da camada oculta
    hidden_spacing_adjustment = 1.5  # Ajusta esse valor conforme necessário para o espaçamento
    for i in range(total_hidden_nodes):
        node_pos[f'Hidden Layer_{i}'] = (1, offset_hidden - i * hidden_spacing_adjustment)

    G.add_nodes_from(node_pos.keys())

    # Inicializando o dicionário para armazenar os rótulos dos pesos das arestas
    edge_labels = {}

    # Adiciona arestas e rótulos de pesos para a primeira camada
    weights_layer1 = model.layer1.weight.data.numpy()
    for i in range(total_input_nodes):
        for j in range(total_hidden_nodes):
            G.add_edge(f'Input Layer_{i}', f'Hidden Layer_{j}')
            weight = weights_layer1[j, i]
            edge_labels[(f'Input Layer_{i}', f'Hidden Layer_{j}')] = f"{weight:.2f}"

    # Adiciona arestas e rótulos de pesos para a segunda camada
    weights_layer2 = model.layer2.weight.data.numpy()
    for i in range(total_hidden_nodes):
        for j in range(total_output_nodes):
            G.add_edge(f'Hidden Layer_{i}', f'Output Layer_{j}')
            weight = weights_layer2[j, i]
            edge_labels[(f'Hidden Layer_{i}', f'Output Layer_{j}')] = f"{weight:.2f}"

    nx.draw(G, pos=node_pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
    nx.draw_networkx_edge_labels(G, pos=node_pos, edge_labels=edge_labels, font_color='red', font_size=10)

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=50, learning_rate=0.01):
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(avaliacoes)
        loss = criterion(output, avaliacoes)
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion):
    avaliacoes_final = avaliacoes_inicial.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))]

    for i, modelo_cliente in enumerate(modelos_clientes):
        # Gerar índices de itens não avaliados
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()
        # Selecionar índices aleatórios para novas avaliações
        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:2]]
        # Gerar novas avaliações aleatórias
        novas_avaliacoes = torch.randint(1, 6, (2,)).float()

        # Atualizar avaliações iniciais com novas avaliações
        avaliacoes_cliente = avaliacoes_inicial[i].clone()
        avaliacoes_cliente[indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes

        print(f"\n=== Treinamento no Cliente {i + 1} ===")
        print(f"Novas Avaliações do Cliente {i + 1}:")
        print_matriz_com_precisao(avaliacoes_cliente.unsqueeze(0))

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=0.01)
        for _ in range(1000):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0))
            loss_cliente = criterion(output_cliente, avaliacoes_cliente.unsqueeze(0))
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0)).squeeze()
        print(f"Novas Recomendações do Cliente {i + 1} após Treinamento Local")
        print_matriz_com_precisao(recomendacoes_cliente.unsqueeze(0))

        print(f"=== Modelo Local Cliente {i+1} após Treinamento Local ===")
        visualize_model(modelo_cliente, f'Modelo Local Cliente {i+1}')

    # Retorna ambos: avaliações finais e os modelos dos clientes
    return avaliacoes_final, modelos_clientes

def agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes):
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes, learning_rate=0.01):
    """
    Atualiza o modelo global com base na média dos gradientes dos modelos locais.

    Args:
    - modelo_global (torch.nn.Module): Modelo global a ser atualizado.
    - modelos_clientes (list): Lista de modelos dos clientes.
    - learning_rate (float): Taxa de aprendizado a ser usada para aplicar os gradientes.
    """
    with torch.no_grad():
        global_params = list(modelo_global.parameters())
        
        # Inicializar uma lista para armazenar a média dos gradientes para cada parâmetro
        gradientes_medios = [torch.zeros_like(param) for param in global_params]
        
        # Calcular a média dos gradientes para cada parâmetro
        for modelo_cliente in modelos_clientes:
            for i, param_cliente in enumerate(modelo_cliente.parameters()):
                if param_cliente.grad is not None:
                    gradientes_medios[i] += param_cliente.grad / len(modelos_clientes)
        
        # Atualizar os parâmetros do modelo global usando a média dos gradientes
        for i, param_global in enumerate(global_params):
            param_global -= learning_rate * gradientes_medios[i]

def mostrar_calculos_agregacao_modelos (modelo_global, modelos_clientes):
    print("\n=== Mostrando Cálculos da Agregação dos Modelos Locais ao Global ===\n")
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = []
            # Extrair os parâmetros correspondentes de todos os modelos dos clientes
            for cliente in modelos_clientes:
                param_cliente = list(cliente.parameters())[i].data
                cliente_params.append(param_cliente)
                print(f"Parâmetros do cliente (camada {i}): {param_cliente}")
            
            cliente_params_stack = torch.stack(cliente_params)
            # Calcular a média dos parâmetros dos clientes e atualizar o modelo global
            param_global.copy_(cliente_params_stack.mean(dim=0))
            
            # Para fins didáticos, vamos calcular e imprimir a média de um parâmetro específico.
            # Exemplo: peso da primeira entrada para o primeiro neurônio da camada escondida.
            if i == 0:  # Assumindo que i == 0 refere-se aos pesos da primeira camada linear.
                media_especifica = cliente_params_stack[:, 0, 0].mean().item()
                print(f"Média dos pesos da primeira entrada para o primeiro neurônio da camada escondida (camada {i}): {media_especifica:.4f}\n")

def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")
    avaliacoes_inicial = torch.tensor([
        [4, 0, 2, 0, 3],
        [0, 3, 0, 4, 0],
        [1, 0, 0, 0, 5],
        [0, 2, 3, 0, 0]
    ], dtype=torch.float32)

    print("=== Matriz de Avaliações Inicial (Servidor) ===")
    print_matriz_com_precisao(avaliacoes_inicial)

    modelo_global = SimpleNN(5, 2, 5)
    criterion = nn.MSELoss() 

    # Chama a função de treinamento
    treinar_modelo_global(modelo_global, avaliacoes_inicial, criterion)

    print("=== Modelo Global Inicial (Servidor) ===")
    visualize_model(modelo_global, 'Modelo Global Inicial')

    with torch.no_grad():
        recomendacoes_inicial = modelo_global(avaliacoes_inicial)

    print("=== Matriz de Recomendações Inicial (Servidor) ===")
    print_matriz_com_precisao(recomendacoes_inicial)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final, modelos_clientes = treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion)

    # Agrega as atualizações dos modelos dos clientes ao modelo global
    agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes)

    mostrar_calculos_agregacao_modelos (modelo_global, modelos_clientes)

    print("\n=== Modelo Global Final (Servidor) ===")
    visualize_model(modelo_global, 'Modelo Global Final')

    with torch.no_grad():
        recomendacoes_final = modelo_global(avaliacoes_inicial)

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL) ===")
    print("=== Matriz de Avaliações Final (Servidor) *** ===")
    print_matriz_com_precisao(avaliacoes_final)

    print("=== Matriz de Recomendações Final (Servidor) ===")
    print_matriz_com_precisao(recomendacoes_final)

    mse_inicial = criterion(recomendacoes_inicial, avaliacoes_inicial).item()
    mse_final = criterion(recomendacoes_final, avaliacoes_final).item()

    print(f"\nMSE Inicial: {mse_inicial:.4f}")
    print(f"MSE Final: {mse_final:.4f}")


if __name__ == "__main__":
    main()
