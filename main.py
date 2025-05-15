import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
# import matplotlib.animation as animation # 애니메이션 사용 안 함
import os

# -----------------------------------------------------------------------------
# 1. 물리 상수 및 시뮬레이션 파라미터 정의 그룹
# -----------------------------------------------------------------------------
G_ACCEL_SI = 9.81      # 중력 가속도 (m/s^2)
RHO_AIR_SI = 1.2       # 공기 밀도 (kg/m^3)
RHO_WATER_SI = 1000.0  # 물 밀도 (kg/m^3)
SIGMA_WATER_SI = 0.072 # 물의 표면 장력 (N/m)
NU_AIR_SI = 1.5e-5     # 공기의 동점성 계수 (m^2/s)

# 시뮬레이션 파라미터
DT_S = 0.0002  # 시뮬레이션 시간 간격 (초)
MAX_SIMULATION_TIME_S = 10.0 # 최대 시뮬레이션 시간 (초)
TERMINAL_VELOCITY_ACCELERATION_THRESHOLD = 1e-5 # 종단 속도 판단 가속도 임계값 (m/s^2)

# 시각화 관련 파라미터
INDENTATION_THRESHOLD_MM = 1.0 
MAX_DIAMETER_FOR_FULL_INDENT_MM = 8.0 

# -----------------------------------------------------------------------------
# 2. 물리 모델 함수 그룹
# -----------------------------------------------------------------------------

def get_drag_coefficient(reynolds_number, aspect_ratio):
    """항력 계수를 계산합니다."""
    if reynolds_number <= 1e-10: 
        return 0.0
    if reynolds_number < 1.0:
        cd_sphere = (24.0 / reynolds_number) * (1.0 + 0.15 * reynolds_number**0.687)
    elif reynolds_number < 800: 
        cd_sphere = (24.0 / reynolds_number) * (1.0 + 0.15 * reynolds_number**0.687) + \
                      (0.42 / (1.0 + 4.25e4 * reynolds_number**-1.16))
    else: 
        cd_sphere = 0.44

    correction_factor = 1.0
    if aspect_ratio > 1.0: 
        correction_factor = 1.0 + 0.2 * (aspect_ratio - 1.0)**0.5
    return cd_sphere * correction_factor

def calculate_aspect_ratio(weber_number, equivalent_diameter_mm):
    """웨버 수에 기반한 변형률을 계산합니다."""
    if weber_number <= 1e-3: 
        return 1.0
    
    # Pruppacher and Pitter (1971) type empirical relation for Dv/Dh
    # (or similar, simplified here for Dh/Dv)
    # 실제로는 직경에 따라 다른 복잡한 관계식을 사용하지만, 여기서는 웨버수에만 의존하는 단순화된 모델을 사용.
    # For small Weber numbers, Dv/Dh approx 1 - (1/9)*We
    # So Dh/Dv approx 1 / (1 - (1/9)*We)
    
    effective_weber_number = min(weber_number, 4.0) # Cap We to prevent extreme aspect ratios from this simple formula

    if effective_weber_number < 1e-3 : 
        return 1.0

    r_vh = 1.0 - (1.0/9.0) * effective_weber_number # Dv/Dh
    
    if r_vh <= 0.1: # Prevent aspect ratio from becoming too extreme
        r_vh = 0.1
        
    ar_horiz_over_vert = 1.0 / r_vh # Dh/Dv
    
    # Cap max aspect ratio (e.g., from observations)
    return min(ar_horiz_over_vert, 2.5) 


def calculate_dimensions_and_area(equivalent_diameter_m, aspect_ratio):
    """등가 직경과 변형률로부터 실제 차원과 단면적을 계산합니다."""
    # V_sphere = (pi/6)*Deq^3 = V_spheroid = (pi/6)*Dh^2*Dv
    # Deq^3 = Dh^2*Dv 
    # aspect_ratio = Dh/Dv  => Dh = aspect_ratio * Dv
    # Deq^3 = (aspect_ratio*Dv)^2 * Dv = aspect_ratio^2 * Dv^3
    # Dv = Deq / (aspect_ratio^(2/3))
    # Dh = aspect_ratio * Dv = Deq * aspect_ratio^(1/3)
    d_vertical_m = equivalent_diameter_m / (aspect_ratio**(2/3))
    d_horizontal_m = equivalent_diameter_m * (aspect_ratio**(1/3))
    cross_sectional_area_m2 = np.pi * (d_horizontal_m / 2)**2 
    return d_horizontal_m, d_vertical_m, cross_sectional_area_m2

# -----------------------------------------------------------------------------
# 3. 단일 빗방울 시뮬레이션 함수 그룹 (복구됨)
# -----------------------------------------------------------------------------
def simulate_single_raindrop(initial_diameter_mm):
    """단일 빗방울의 낙하를 시뮬레이션합니다."""
    diameter_m = initial_diameter_mm / 1000.0
    volume_m3 = (4/3) * np.pi * (diameter_m/2)**3
    mass_kg = RHO_WATER_SI * volume_m3
    
    velocity_y_m_s = 0.0
    current_time_s = 0.0
    current_weber_number = 0.0 # 초기 웨버 수는 0
    
    # evolution_states: (시간, 수평직경_mm, 수직직경_mm, 단면적_m2)
    evolution_states = [] 

    # 초기 상태 (T0) 저장
    initial_equivalent_diameter_m = (6 * mass_kg / (RHO_WATER_SI * np.pi))**(1/3)
    # T0에서는 속도가 0이므로 웨버 수도 0, 따라서 변형률은 1 (구형)
    initial_aspect_ratio = calculate_aspect_ratio(0.0, initial_equivalent_diameter_m * 1000) 
    d_h_initial_m, d_v_initial_m, area_initial_m2 = calculate_dimensions_and_area(
        initial_equivalent_diameter_m, initial_aspect_ratio
    )
    evolution_states.append((0.0, d_h_initial_m * 1000, d_v_initial_m * 1000, area_initial_m2))

    print(f"\n--- 직경 {initial_diameter_mm:.2f} mm 빗방울 시뮬레이션 시작 ---")

    for step in range(int(MAX_SIMULATION_TIME_S / DT_S)):
        equivalent_diameter_m = initial_equivalent_diameter_m # 질량 보존으로 등가 직경은 일정
        
        # 현재 웨버 수(이전 스텝의 속도로 계산됨)를 기반으로 현재 변형률 계산
        current_aspect_ratio = calculate_aspect_ratio(current_weber_number, equivalent_diameter_m * 1000)
        
        # 중력 계산
        force_gravity_N = mass_kg * G_ACCEL_SI
        
        # 항력 계산
        force_drag_N = 0.0
        if abs(velocity_y_m_s) > 1e-9:
            reynolds_number = abs(velocity_y_m_s) * equivalent_diameter_m / NU_AIR_SI
            drag_coeff = get_drag_coefficient(reynolds_number, current_aspect_ratio)
            
            # 항력 계산 시 사용될 차원 및 단면적 (현재 변형률 기준)
            d_h_drag_m, d_v_drag_m, area_drag_m2 = calculate_dimensions_and_area(equivalent_diameter_m, current_aspect_ratio)
            drag_magnitude = 0.5 * RHO_AIR_SI * velocity_y_m_s**2 * drag_coeff * area_drag_m2
            force_drag_N = -np.sign(velocity_y_m_s) * drag_magnitude

        # 총 힘 및 가속도
        total_force_N = force_gravity_N + force_drag_N
        acceleration_y_m_s2 = total_force_N / mass_kg
        
        # 속도 및 시간 업데이트
        velocity_y_m_s += acceleration_y_m_s2 * DT_S
        current_time_s += DT_S

        # 다음 스텝의 변형률 계산을 위해 현재 속도로 웨버 수 업데이트
        if abs(velocity_y_m_s) > 1e-9:
            current_weber_number = RHO_AIR_SI * velocity_y_m_s**2 * equivalent_diameter_m / SIGMA_WATER_SI
        else:
            current_weber_number = 0.0 # 속도가 0이면 웨버 수도 0

        # 상태 저장 (업데이트된 웨버 수와 그에 따른 변형률 기준)
        aspect_ratio_for_state = calculate_aspect_ratio(current_weber_number, equivalent_diameter_m * 1000)
        d_h_state_m, d_v_state_m, area_state_m2 = calculate_dimensions_and_area(equivalent_diameter_m, aspect_ratio_for_state)
        
        if current_time_s > 1e-9: # T0 이후의 상태만 추가 (T0는 이미 추가됨)
            evolution_states.append((current_time_s, d_h_state_m * 1000, d_v_state_m * 1000, area_state_m2))

        # 종료 조건
        if abs(acceleration_y_m_s2) < TERMINAL_VELOCITY_ACCELERATION_THRESHOLD and current_time_s > 0.1:
            # print(f"  시간: {current_time_s:.3f}s - 종단 속도 도달.")
            break
        if current_time_s >= MAX_SIMULATION_TIME_S:
            # print(f"  시간: {current_time_s:.3f}s - 최대 시뮬레이션 시간 도달.")
            break
            
    terminal_velocity_mps = abs(velocity_y_m_s)
    time_to_terminal_s = current_time_s
    
    # 종단 상태에서의 최종 값들
    weber_at_terminal = RHO_AIR_SI * terminal_velocity_mps**2 * equivalent_diameter_m / SIGMA_WATER_SI if terminal_velocity_mps > 1e-9 else 0.0
    aspect_ratio_at_terminal = calculate_aspect_ratio(weber_at_terminal, equivalent_diameter_m * 1000)
    d_h_terminal_m, d_v_terminal_m, area_terminal_m2 = calculate_dimensions_and_area(equivalent_diameter_m, aspect_ratio_at_terminal)
    reynolds_at_terminal = abs(terminal_velocity_mps) * equivalent_diameter_m / NU_AIR_SI if terminal_velocity_mps > 1e-9 else 0.0

    # T/2 시점 데이터 추출
    shape_at_T_half = {"d_horizontal_mm": None, "d_vertical_mm": None, "area_m2": None}
    if evolution_states and len(evolution_states) > 1:
        all_times = np.array([state[0] for state in evolution_states])
        effective_terminal_time = all_times[-1]
        time_T_half = effective_terminal_time / 10.0
        
        valid_indices = np.where(all_times >= 0)[0] # 모든 시간은 0 이상이어야 함
        if len(valid_indices) > 0:
            idx_T_half = valid_indices[np.abs(all_times[valid_indices] - time_T_half).argmin()]
            if idx_T_half < len(evolution_states): # 인덱스 범위 확인
                shape_at_T_half["d_horizontal_mm"] = evolution_states[idx_T_half][1]
                shape_at_T_half["d_vertical_mm"] = evolution_states[idx_T_half][2]
                shape_at_T_half["area_m2"] = evolution_states[idx_T_half][3]
    
    return {
        "initial_diameter_mm": initial_diameter_mm,
        "terminal_velocity_mps": terminal_velocity_mps,
        "aspect_ratio_at_terminal": aspect_ratio_at_terminal,
        "d_horizontal_mm_at_terminal": d_h_terminal_m * 1000,
        "d_vertical_mm_at_terminal": d_v_terminal_m * 1000,
        "cross_sectional_area_m2_at_terminal": area_terminal_m2,
        "time_to_terminal_s": time_to_terminal_s,
        "reynolds_at_terminal": reynolds_at_terminal,
        "weber_at_terminal": weber_at_terminal,
        "shape_at_T_half": shape_at_T_half, # T/4는 제거하고 T/2만 유지
        "evolution_states": evolution_states,
    }

# -----------------------------------------------------------------------------
# 4. 결과 플로팅 함수 그룹
# -----------------------------------------------------------------------------

def plot_results(results_list):
    """시뮬레이션 결과의 정량적 데이터를 플로팅합니다."""
    if not results_list:
        # print("플로팅할 결과 데이터가 없습니다 (plot_results).") # 사용자에게 불필요할 수 있음
        return
    
    valid_results = [r for r in results_list if isinstance(r, dict) and 
                     all(k in r for k in ["initial_diameter_mm", "terminal_velocity_mps", 
                                          "aspect_ratio_at_terminal", "cross_sectional_area_m2_at_terminal"])]
    if not valid_results:
        # print("유효한 결과 데이터가 없어 플로팅을 건너<0xEB><0xA9><0xB Skip>니다 (plot_results).")
        return

    diameters_mm = [r["initial_diameter_mm"] for r in valid_results]
    terminal_velocities = [r["terminal_velocity_mps"] for r in valid_results]
    aspect_ratios = [r["aspect_ratio_at_terminal"] for r in valid_results]
    cross_sectional_areas = [r["cross_sectional_area_m2_at_terminal"] for r in valid_results]

    plt.rcParams['font.family'] = [
        'Malgun Gothic',  # 윈도우
        'AppleGothic',    # macOS
        'NanumGothic',    # 리눅스에 나눔글꼴 설치되어 있을 때
        'DejaVu Sans',    # 기본 영문/숫자 폰트
        'Arial',
        'Helvetica' 
    ]
    plt.rcParams['axes.unicode_minus'] = False  
    
    '''try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # print("'맑은 고딕' 폰트를 찾을 수 없거나 설정에 실패했습니다. 기본 산세리프 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'AppleGothic', 'NanumGothic', 'Bitstream Vera Sans']
        plt.rcParams['axes.unicode_minus'] = False'''

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("빗방울 크기별 종단 특성 (정량적 분석)", fontsize=16)

    axs[0].plot(diameters_mm, terminal_velocities, marker='o', linestyle='-', color='blue')
    axs[0].set_ylabel("종단 속도 (m/s)")
    axs[0].grid(True)
    axs[0].set_title("빗방울 크기 대 종단 속도")

    axs[1].plot(diameters_mm, aspect_ratios, marker='s', linestyle='--', color='green')
    axs[1].set_ylabel("변형률 (D_h/D_v)")
    axs[1].grid(True)
    axs[1].set_title("빗방울 크기 대 종단 속도에서의 변형률")

    axs[2].plot(diameters_mm, cross_sectional_areas, marker='^', linestyle=':', color='red')
    axs[2].set_xlabel("빗방울 초기 직경 (mm)")
    axs[2].set_ylabel("단면적 (m^2)")
    axs[2].grid(True)
    axs[2].set_title("빗방울 크기 대 종단 속도에서의 수평 단면적")
    if any(a > 1e-9 for a in cross_sectional_areas if isinstance(a, (int, float))):
        try:
            axs[2].set_yscale('log')
        except ValueError:
            axs[2].set_yscale('linear')

    plt.tight_layout(rect=[0, 0, 1, 0.93])


def plot_raindrop_shapes(results_list, target_diameters_mm=None,
                         indentation_threshold_param=INDENTATION_THRESHOLD_MM,
                         max_diameter_for_full_indent_param=MAX_DIAMETER_FOR_FULL_INDENT_MM):
    """선택된 빗방울의 특정 시점(T0, T/2, 종단)에서의 단면 모양을 플로팅합니다."""
    if not results_list:
        # print("플로팅할 결과 데이터가 없습니다 (plot_raindrop_shapes).")
        return

    selected_results_dict = {}
    if target_diameters_mm:
        for target_d in target_diameters_mm:
            valid_results = [r for r in results_list if isinstance(r, dict) and "initial_diameter_mm" in r]
            if not valid_results: continue
            closest_result = min(valid_results, key=lambda r: abs(r["initial_diameter_mm"] - target_d))
            if abs(closest_result["initial_diameter_mm"] - target_d) < 0.5 :
                selected_results_dict[closest_result["initial_diameter_mm"]] = closest_result
    else: 
        valid_results = [r for r in results_list if isinstance(r, dict) and "initial_diameter_mm" in r]
        if len(valid_results) >= 1:
            num_to_select = min(len(valid_results), 3)
            indices = np.linspace(0, len(valid_results) - 1, num_to_select, dtype=int)
            for i in indices:
                 selected_results_dict[valid_results[i]["initial_diameter_mm"]] = valid_results[i]

    selected_results = list(selected_results_dict.values())
    selected_results.sort(key=lambda r: r["initial_diameter_mm"])

    if not selected_results:
        # print("시각화할 대상 빗방울을 찾지 못했습니다 (plot_raindrop_shapes).")
        return

    num_target_drops = len(selected_results)
    fig_shapes, axs_shapes_flat = plt.subplots(num_target_drops, 3,
                                               figsize=(3 * 4.0, num_target_drops * 4.5),
                                               squeeze=False,
                                               constrained_layout=True)
    fig_shapes.suptitle("빗방울 크기별 단면 모양 변화 (T0, T/2, 종단 시점)", fontsize=16)

    max_dim_overall = 0 
    for r_val in selected_results: 
        if not isinstance(r_val, dict) : continue 
        max_dim_overall = max(max_dim_overall, r_val.get("d_horizontal_mm_at_terminal",0), r_val.get("d_vertical_mm_at_terminal",0))
        evolution_states = r_val.get("evolution_states", [])
        if evolution_states and len(evolution_states) > 0 and len(evolution_states[0]) > 2:
             max_dim_overall = max(max_dim_overall, evolution_states[0][1], evolution_states[0][2]) 
        
        shape_at_T_half_data = r_val.get("shape_at_T_half",{})
        if isinstance(shape_at_T_half_data, dict) and shape_at_T_half_data.get("d_horizontal_mm") is not None:
            max_dim_overall = max(max_dim_overall, shape_at_T_half_data.get("d_horizontal_mm",0), shape_at_T_half_data.get("d_vertical_mm",0))
    if max_dim_overall <= 1e-6 : max_dim_overall = 1.0


    for row_idx, result_item in enumerate(selected_results): 
        if not isinstance(result_item, dict) : continue 
        initial_d_mm = result_item.get("initial_diameter_mm", 0)
        
        shape_at_T0 = {"d_horizontal_mm": None, "d_vertical_mm": None, "area_m2": None}
        evolution_states = result_item.get("evolution_states", [])
        if evolution_states and len(evolution_states) > 0 and len(evolution_states[0]) > 3: 
            t0_state = evolution_states[0]
            shape_at_T0["d_horizontal_mm"] = t0_state[1]
            shape_at_T0["d_vertical_mm"] = t0_state[2]
            shape_at_T0["area_m2"] = t0_state[3]
        
        shapes_to_plot = [
            (shape_at_T0, "T0 (초기) 시점"), 
            (result_item.get("shape_at_T_half",{}), "T/2 시점"), 
            ({"d_horizontal_mm": result_item.get("d_horizontal_mm_at_terminal"),
              "d_vertical_mm": result_item.get("d_vertical_mm_at_terminal"),
              "area_m2": result_item.get("cross_sectional_area_m2_at_terminal")},
             "종단 (T) 시점")
        ]

        for col_idx, (shape_data, time_label) in enumerate(shapes_to_plot):
            ax_s = axs_shapes_flat[row_idx, col_idx]
            if not isinstance(shape_data, dict) or \
               shape_data.get("d_horizontal_mm") is None or \
               shape_data.get("d_vertical_mm") is None or \
               shape_data.get("d_horizontal_mm", 0) <= 1e-6 or \
               shape_data.get("d_vertical_mm", 0) <= 1e-6:
                ax_s.text(0.5, 0.5, "데이터 없음\n또는 매우 작음", ha='center', va='center', fontsize=9, color='gray')
                ax_s.set_title(f"초기 {initial_d_mm:.1f}mm - {time_label}", fontsize=10)
                ax_s.set_aspect('equal', adjustable='box'); ax_s.set_xticks([]); ax_s.set_yticks([])
                continue

            d_horiz_mm = shape_data["d_horizontal_mm"]
            d_vert_mm = shape_data["d_vertical_mm"]
            current_area_m2 = shape_data.get("area_m2", 0)
            
            local_max_indent_fraction = 0.33
            local_indent_scaling_exponent = 1.0
            local_indent_shape_exponent = 2.0

            if initial_d_mm < indentation_threshold_param or time_label == "T0 (초기) 시점":
            # if initial_d_mm < indentation_threshold_param or time_label != "종단 (T) 시점":
                shape_patch = patches.Ellipse(xy=(0, 0), width=d_horiz_mm, height=d_vert_mm, angle=0,
                                              facecolor='skyblue', edgecolor='blue', alpha=0.7)
                ax_s.add_patch(shape_patch)
            else: 
                num_poly_points_static = 120; rx = d_horiz_mm / 2.0; ry = d_vert_mm / 2.0
                thetas = np.linspace(0, 2 * np.pi, num_poly_points_static, endpoint=False)
                polygon_points = []

                denominator_for_norm = (max_diameter_for_full_indent_param - indentation_threshold_param)
                if abs(denominator_for_norm) <= 1e-6: 
                    normalized_size_factor = 1.0 if initial_d_mm >= indentation_threshold_param else 0.0
                else:
                    normalized_size_factor = min(1.0, max(0.0, (initial_d_mm - indentation_threshold_param) / denominator_for_norm ))

                max_indent_abs = ry * local_max_indent_fraction * normalized_size_factor**local_indent_scaling_exponent

                for t in thetas:
                    x = rx * np.cos(t)
                    y_ellipse = ry * np.sin(t)
                    y_final = y_ellipse
                    if y_ellipse < 0: 
                        indent_profile_factor = np.sin(t - np.pi)**local_indent_shape_exponent
                        y_final = y_ellipse + max_indent_abs * indent_profile_factor
                    polygon_points.append((x, y_final))

                shape_patch = patches.Polygon(polygon_points, closed=True,
                                              facecolor='lightblue', edgecolor='darkblue', alpha=0.75)
                ax_s.add_patch(shape_patch)

            title_text = f"초기(Initial) {initial_d_mm:.1f}mm - {time_label}\n"
            title_text += f"(H:{d_horiz_mm:.2f} x V:{d_vert_mm:.2f} mm)\n"
            title_text += f"단면적(Area): {current_area_m2:.2e} m²"
            ax_s.set_title(title_text, fontsize=8)

            axis_limit = max_dim_overall * 1.2 / 2.0 
            ax_s.set_xlim(-axis_limit, axis_limit)
            ax_s.set_ylim(-axis_limit, axis_limit)
            ax_s.set_aspect('equal', adjustable='box')

            if row_idx == num_target_drops -1 : ax_s.set_xlabel("수평 (Horizontal) (mm)")
            else: ax_s.set_xticklabels([])
            if col_idx == 0: ax_s.set_ylabel("수직 (Vertical) (mm)")
            else: ax_s.set_yticklabels([])
            ax_s.grid(True, linestyle=':', alpha=0.5)

    # plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    # 겹침 방지를 위해 서브플롯 여백 직접 조절
    fig_shapes.subplots_adjust(
        top=0.88,    # 전체 제목 위쪽 여백
        bottom=0.05, # 아래쪽 여백
        left=0.05,   # 왼쪽 여백
        right=0.95,  # 오른쪽 여백
        hspace=0.6,  # 행 간격
        wspace=0.4   # 열 간격
    )

def plot_overlay_terminal_shapes(diameters_mm,
                                 indentation_threshold_mm=INDENTATION_THRESHOLD_MM,
                                 max_indent_diameter_mm=MAX_DIAMETER_FOR_FULL_INDENT_MM):
    """
    여러 초기 직경의 빗방울을 한 축에 겹쳐서, 종단 시점에 얻은 단면 모양 윤곽을 그립니다.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("수평 (Horizontal) (mm)")
    ax.set_ylabel("수직 (Vertical) (mm)")
    ax.set_title("빗방울 크기별 단면 모양 변화 (종단 시점)")

    # 색상 맵
    cmap = plt.get_cmap('viridis', len(diameters_mm))
    for i, d0 in enumerate(diameters_mm):
        res = simulate_single_raindrop(d0)
        dh = res["d_horizontal_mm_at_terminal"]
        dv = res["d_vertical_mm_at_terminal"]

        # 폴리곤 생성
        rx, ry = dh/2, dv/2
        thetas = np.linspace(0, 2*np.pi, 200)
        xs, ys = rx*np.cos(thetas), ry*np.sin(thetas)

        if d0 >= indentation_threshold_mm:
            # 아래쪽에 움푹 패인 모양으로 변형
            indent_mag = ry * 0.33 * min(1, (d0 - indentation_threshold_mm)/(max_indent_diameter_mm - indentation_threshold_mm))
            ys = np.where(
                ys < 0,
                ys + indent_mag * np.sin(thetas - np.pi)**2,
                ys
            )

        # 윤곽선만
        ax.plot(xs, ys, color=cmap(i), label=f"{d0} mm")

        # 오른쪽 가장 바깥 점에 라벨
        # 시계 방향으로 10° 안쪽에 표시
        ang = -10 * np.pi/180
        x_label, y_label = rx*np.cos(ang), (ry*np.sin(ang) if d0 < indentation_threshold_mm else
                                            (ry*np.sin(ang) + indent_mag*np.sin(ang-np.pi)**2))
        ax.text(x_label*1.05, y_label*1.05, f"{d0}", color=cmap(i),
                fontsize= 9, va='center', ha='center')

    ax.legend(title="초기 지름", loc='upper right', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.show()


# -----------------------------------------------------------------------------
# 애니메이션 관련 함수 및 로직은 제거되었습니다.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. 메인 실행 로직 (시뮬레이션 복구)
# -----------------------------------------------------------------------------

def main():
    """
    메인 실행 함수.
    빗방울 시뮬레이션을 실행하고, 그 결과를 바탕으로 정적 플롯을 생성합니다.
    """
    
    raindrop_initial_diameters_mm = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] 
    simulation_results = []

    print("빗방울 시뮬레이션을 시작합니다...")
    for diameter_mm in raindrop_initial_diameters_mm:
        result = simulate_single_raindrop(diameter_mm)
        if result: 
            simulation_results.append(result)
    print("모든 시뮬레이션 완료.")

    if not simulation_results:
        print("시뮬레이션 결과가 없어 플로팅을 실행할 수 없습니다.")
        return

    print("\n정적 플롯을 생성합니다...")
    plot_results(simulation_results)
    plot_raindrop_shapes(simulation_results, target_diameters_mm=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    print("플롯 창을 화면에 표시합니다. 모든 창을 닫으면 프로그램이 종료됩니다.")
    plt.show() 
    
    # 1mm, 2mm, …, 6mm 빗방울을 겹쳐서 그리기
    plot_overlay_terminal_shapes([1, 2, 3, 4, 5, 6])

    print("\n모든 작업 완료.")

if __name__ == '__main__':
    main()
