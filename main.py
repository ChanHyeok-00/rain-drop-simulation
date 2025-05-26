import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os 
from matplotlib.ticker import ScalarFormatter


# -----------------------------------------------------------------------------
# 1. 물리 상수 및 시뮬레이션 파라미터 정의 그룹
# -----------------------------------------------------------------------------
G_ACCEL_SI = 9.81      # 중력 가속도 (m/s^2)
RHO_AIR_SI = 1.2       # 공기 밀도 (kg/m^3)
RHO_WATER_SI = 1000.0  # 물 밀도 (kg/m^3)
SIGMA_WATER_SI = 0.072 # 물의 표면 장력 (N/m)
NU_AIR_SI = 1.5e-5     # 공기의 동점성 계수 (m^2/s) (점성계수 / 공기밀도)

DT_S = 0.0002                                       # 시뮬레이션 시간 간격 (초)
MAX_SIMULATION_TIME_S = 10.0                        # 최대 시뮬레이션 시간 (초)
TERMINAL_VELOCITY_ACCELERATION_THRESHOLD = 1e-5     # 종단 속도 판단 가속도 임계값 (m/s^2)
INDENTATION_THRESHOLD_MM = 1.0                      # 이 직경(mm) 이상부터 함몰 표현 시도 (plot_overlay_terminal_shapes에서 사용)
MAX_DIAMETER_FOR_FULL_INDENT_MM = 6.0               # 이 직경(mm)에서 함몰 정도가 최대로 표현 (plot_overlay_terminal_shapes에서 사용)
TARGET_ANIMATION_FRAMES = 40                        # 애니메이션 목표 프레임 수
WEBER_CRITICAL = 12.0                               # 분열 임계 웨버 수, 시뮬레이션 종료 판단 기준

# 함몰 여부를 결정하는 웨버 수 (create_raindrop_patch에서 사용)
WEBER_INDENT_START_THRESHOLD = 0.25                 # 지름이 1mm일 때의 웨버 수
WEBER_INDENT_FULL_THRESHOLD = 7.5                   # 지름이 6mm일 때의 웨버 수


# -----------------------------------------------------------------------------
# 2. 물리 모델 함수 그룹
# -----------------------------------------------------------------------------
def get_drag_coefficient(reynolds_number, aspect_ratio):
    # if reynolds_number <= 1e-6: return 0.0
    if reynolds_number < 1.0: cd_sphere = 24.0 / reynolds_number
    elif reynolds_number < 800: cd_sphere = (24.0 / reynolds_number) * (1.0 + 0.15 * reynolds_number**0.687) + (0.42 / (1.0 + 4.25e4 * reynolds_number**-1.16))
    else: cd_sphere = 0.44
    correction_factor = 1.0
    if aspect_ratio > 1.0: correction_factor = 1.0 + 0.2 * (aspect_ratio - 1.0)**0.5 
    return cd_sphere * correction_factor

def calculate_aspect_ratio(weber_number, equivalent_diameter_mm):
    if equivalent_diameter_mm < 1.0: return 1.0
    k_weber_effect = 0.07  # 웨버 수가 변형률에 미치는 경험적인 상수
    current_we = max(0, weber_number) # 웨버 수를 양수로 제한
    ar = 1.0 + k_weber_effect * current_we # 변형률
    MAX_ASPECT_RATIO = 2.5 # 너무 심하게 찌그러지는 것을 방지하기 위해 설정한 변형률의 최댓값
    ar = min(ar, MAX_ASPECT_RATIO) 
    return ar

# 빗방울의 수직 직경, 수평 직경, 수평 단면적 계산
def calculate_dimensions_and_area(equivalent_diameter_m, aspect_ratio):
    d_vertical_m = equivalent_diameter_m / (aspect_ratio**(2/3))
    d_horizontal_m = equivalent_diameter_m * (aspect_ratio**(1/3))
    cross_sectional_area_m2 = np.pi * (d_horizontal_m / 2)**2
    return d_horizontal_m, d_vertical_m, cross_sectional_area_m2


# -----------------------------------------------------------------------------
# 3. 단일 빗방울 시뮬레이션 함수 그룹
# -----------------------------------------------------------------------------
def simulate_single_raindrop(initial_diameter_mm):
    diameter_m = initial_diameter_mm / 1000.0; volume_m3 = (4/3) * np.pi * (diameter_m/2)**3; mass_kg = RHO_WATER_SI * volume_m3
    position_y_m = 0.0; velocity_y_m_s = 0.0; current_time_s = 0.0; current_aspect_ratio = 1.0; current_reynolds_number = 0.0; current_weber_number = 0.0
    evolution_states = [(0.0, initial_diameter_mm, initial_diameter_mm, 0.0, 0.0)] 
    
    for step in range(int(MAX_SIMULATION_TIME_S / DT_S)): 
        equivalent_diameter_m = (6 * mass_kg / (RHO_WATER_SI * np.pi))**(1/3); force_gravity_N = mass_kg * G_ACCEL_SI; force_drag_N = 0.0
        if abs(velocity_y_m_s) > 1e-6:
            current_reynolds_number = abs(velocity_y_m_s) * equivalent_diameter_m / NU_AIR_SI; cd = get_drag_coefficient(current_reynolds_number, current_aspect_ratio) 
            _, _, area_m2 = calculate_dimensions_and_area(equivalent_diameter_m, current_aspect_ratio); drag_magnitude = 0.5 * RHO_AIR_SI * velocity_y_m_s**2 * cd * area_m2
            force_drag_N = -np.sign(velocity_y_m_s) * drag_magnitude
        total_force_N = force_gravity_N + force_drag_N; acceleration_y_m_s2 = total_force_N / mass_kg
        velocity_y_m_s += acceleration_y_m_s2 * DT_S; position_y_m += velocity_y_m_s * DT_S; current_time_s += DT_S
        if abs(velocity_y_m_s) > 1e-6:
            current_weber_number = RHO_AIR_SI * velocity_y_m_s**2 * equivalent_diameter_m / SIGMA_WATER_SI; current_aspect_ratio = calculate_aspect_ratio(current_weber_number, equivalent_diameter_m * 1000)
        else: current_weber_number = 0.0; current_aspect_ratio = 1.0 
        d_h_final_step_m, d_v_final_step_m, _ = calculate_dimensions_and_area(equivalent_diameter_m, current_aspect_ratio)
        evolution_states.append((current_time_s, d_h_final_step_m * 1000, d_v_final_step_m * 1000, position_y_m, current_weber_number))
        if abs(acceleration_y_m_s2) < TERMINAL_VELOCITY_ACCELERATION_THRESHOLD: break
        if current_weber_number > WEBER_CRITICAL: break
        if current_time_s >= MAX_SIMULATION_TIME_S: break
            
    terminal_velocity_mps = abs(velocity_y_m_s); time_to_terminal_s = current_time_s; final_aspect_ratio = current_aspect_ratio 
    final_equivalent_diameter_m = (6 * mass_kg / (RHO_WATER_SI * np.pi))**(1/3)
    d_horiz_m, d_vert_m, final_area_m2 = calculate_dimensions_and_area(final_equivalent_diameter_m, final_aspect_ratio)
    shape_at_T_quarter = {"d_horizontal_mm": None, "d_vertical_mm": None}; shape_at_T_half = {"d_horizontal_mm": None, "d_vertical_mm": None}
    all_times = np.array([state[0] for state in evolution_states]); time_T_quarter = time_to_terminal_s / 4.0; idx_T_quarter = np.abs(all_times - time_T_quarter).argmin()
    shape_at_T_quarter["d_horizontal_mm"] = evolution_states[idx_T_quarter][1]; shape_at_T_quarter["d_vertical_mm"] = evolution_states[idx_T_quarter][2]
    time_T_half = time_to_terminal_s / 2.0; idx_T_half = np.abs(all_times - time_T_half).argmin()
    shape_at_T_half["d_horizontal_mm"] = evolution_states[idx_T_half][1]; shape_at_T_half["d_vertical_mm"] = evolution_states[idx_T_half][2]
    
    return {"initial_diameter_mm": initial_diameter_mm, "terminal_velocity_mps": terminal_velocity_mps, "aspect_ratio_at_terminal": final_aspect_ratio, "d_horizontal_mm_at_terminal": d_horiz_m * 1000, "d_vertical_mm_at_terminal": d_vert_m * 1000, "cross_sectional_area_m2_at_terminal": final_area_m2, "time_to_terminal_s": time_to_terminal_s, "reynolds_at_terminal": current_reynolds_number, "weber_at_terminal": current_weber_number, "shape_at_T_quarter": shape_at_T_quarter, "shape_at_T_half": shape_at_T_half, "evolution_states": evolution_states}

def plot_overlay_terminal_shapes(simulation_results, indentation_threshold_param, max_diameter_for_full_indent_param):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Horizontal (mm)")
    ax.set_ylabel("Vertical (mm)")
    ax.set_title("Terminal Shapes vs. Initial Diameter")

    cmap = plt.get_cmap('viridis', len(simulation_results))
    
    for i, res in enumerate(simulation_results):
        d0 = res["initial_diameter_mm"]
        dh = res["d_horizontal_mm_at_terminal"]
        dv = res["d_vertical_mm_at_terminal"]

        if dh is None or dv is None or dh <= 0 or dv <= 0:
            continue

        rx, ry = dh/2, dv/2
        thetas = np.linspace(0, 2*np.pi, 200)
        xs, ys = rx*np.cos(thetas), ry*np.sin(thetas)

        # Indentation logic based on diameter threshold (using passed parameters)
        if d0 >= indentation_threshold_param:
            indent_mag_denom = (max_diameter_for_full_indent_param - indentation_threshold_param)
            indent_mag_denom = indent_mag_denom if indent_mag_denom > 1e-6 else 1.0
            
            indent_mag = ry * 0.33 * min(1, (d0 - indentation_threshold_param) / indent_mag_denom)
            ys = np.where(
                ys < 0,
                ys + indent_mag * np.sin(thetas - np.pi)**2,
                ys
            )

        ax.plot(xs, ys, color=cmap(i), label=f"{d0} mm")

        # Text label logic based on diameter threshold
        ang = -10 * np.pi/180
        x_label_base, y_label_base = rx*np.cos(ang), ry*np.sin(ang)
        if d0 >= indentation_threshold_param and y_label_base < 0:
             y_label_final = y_label_base + indent_mag * np.sin(ang - np.pi)**2
        else:
             y_label_final = y_label_base
        ax.text(x_label_base*1.05, y_label_final*1.05, f"{d0}", color=cmap(i),
                fontsize= 9, va='center', ha='center')

    ax.legend(title="Initial Diameter", loc='upper right', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.show()


# -----------------------------------------------------------------------------
# 4. 메인 실행 로직 및 결과 플로팅 그룹
# -----------------------------------------------------------------------------
def create_raindrop_patch(d_h_mm, d_v_mm, weber_number):
    """Generates a Matplotlib Patch for a raindrop based on its dimensions and Weber number."""
    normalized_we_for_indent = 0.0
    if WEBER_INDENT_FULL_THRESHOLD > WEBER_INDENT_START_THRESHOLD:
        normalized_we_for_indent = min(1.0, max(0.0, (weber_number - WEBER_INDENT_START_THRESHOLD) / (WEBER_INDENT_FULL_THRESHOLD - WEBER_INDENT_START_THRESHOLD)))
    elif weber_number >= WEBER_INDENT_START_THRESHOLD:
        normalized_we_for_indent = 1.0
    max_indent_fraction = 0.35
    max_indent_abs = (d_v_mm / 2.0) * max_indent_fraction * normalized_we_for_indent
    if max_indent_abs < 0.001: # 거의 함몰이 없으면 타원으로 처리
        return patches.Ellipse(xy=(0, 0), width=d_h_mm, height=d_v_mm, angle=0)
    else: # 함몰이 있으면 다각형으로 모양 생성
        num_poly_points = 120; rx, ry = d_h_mm / 2.0, d_v_mm / 2.0
        thetas = np.linspace(0, 2 * np.pi, num_poly_points, endpoint=False)
        polygon_points = []
        for t in thetas:
            x = rx * np.cos(t); y_ellipse = ry * np.sin(t); y_final = y_ellipse
            if y_ellipse < 0: # 아랫부분만 함몰 적용
                indent_shape_exponent = 2.0; indent_profile_factor = np.sin(t - np.pi)**indent_shape_exponent
                y_final = y_ellipse + max_indent_abs * indent_profile_factor
            polygon_points.append((x, y_final))
        return patches.Polygon(polygon_points, closed=True)

def plot_results(results_list):
    diameters_mm = [r["initial_diameter_mm"] for r in results_list]; terminal_velocities = [r["terminal_velocity_mps"] for r in results_list]; aspect_ratios = [r["aspect_ratio_at_terminal"] for r in results_list]; cross_sectional_areas = [r["cross_sectional_area_m2_at_terminal"] for r in results_list]
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Terminal Characteristics vs. Raindrop Size", fontsize=16)
    axs[0].plot(diameters_mm, terminal_velocities, marker='o', linestyle='-', color='blue'); axs[0].set_ylabel("Terminal Velocity (m/s)"); axs[0].grid(True); axs[0].set_title("Terminal Velocity vs. Raindrop Size")
    axs[1].plot(diameters_mm, aspect_ratios, marker='s', linestyle='--', color='green'); axs[1].set_ylabel("Aspect Ratio (D_horiz/D_vert)"); axs[1].grid(True); axs[1].set_title("Aspect Ratio at Terminal Velocity vs. Raindrop Size")
    axs[2].plot(diameters_mm, cross_sectional_areas, marker='^', linestyle=':', color='red'); axs[2].set_xlabel("Initial Diameter (mm)"); axs[2].set_ylabel("Cross-sectional Area (m^2)"); axs[2].grid(True); axs[2].set_title("Cross-sectional Area at Terminal Velocity vs. Raindrop Size")
    formatter = ScalarFormatter(useMathText=False); formatter.set_scientific(True); formatter.set_powerlimits((-1,1)); axs[2].yaxis.set_major_formatter(formatter)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_raindrop_shapes(results_list, target_diameters_mm=None):
    selected_results_dict = {} 
    if target_diameters_mm:
        for target_d in target_diameters_mm:
            closest_result = min(results_list, key=lambda r: abs(r["initial_diameter_mm"] - target_d)); selected_results_dict[closest_result["initial_diameter_mm"]] = closest_result
    else: 
        num_to_select = min(len(results_list), 3); indices = np.linspace(0, len(results_list) - 1, num_to_select, dtype=int)
        for i in indices: selected_results_dict[results_list[i]["initial_diameter_mm"]] = results_list[i]
    
    selected_results = list(selected_results_dict.values()); selected_results.sort(key=lambda r: r["initial_diameter_mm"]) 
    
    num_target_drops = len(selected_results); fig_shapes, axs_shapes_flat = plt.subplots(num_target_drops, 3, figsize=(3 * 4.0, num_target_drops * 4.5), squeeze=False) 
    fig_shapes.suptitle("Cross-sectional Shape Evolution (t=0, t=T/8, Terminal)", fontsize=16)

    fixed_axis_limit = 5.0 

    for row_idx, result in enumerate(selected_results):
        initial_d_mm = result["initial_diameter_mm"]; all_states = result["evolution_states"]; all_times = np.array([state[0] for state in all_states])
        
        shape_data_t0 = {"d_horizontal_mm": initial_d_mm, "d_vertical_mm": initial_d_mm}
        weber_at_t0 = 0.0

        time_to_terminal = result["time_to_terminal_s"]
        time_T_tenth = time_to_terminal / 8.0
        idx_T_tenth = np.abs(all_times - time_T_tenth).argmin()
        
        d_h_at_T_tenth = all_states[idx_T_tenth][1]
        d_v_at_T_tenth = all_states[idx_T_tenth][2]
        shape_data_T_tenth = {"d_horizontal_mm": d_h_at_T_tenth, "d_vertical_mm": d_v_at_T_tenth}
        weber_at_T_tenth = all_states[idx_T_tenth][4]

        shape_data_T = {"d_horizontal_mm": result["d_horizontal_mm_at_terminal"], 
                        "d_vertical_mm": result["d_vertical_mm_at_terminal"]}
        weber_at_T = result["weber_at_terminal"]

        shapes_to_plot = [
            (shape_data_t0, "t = 0 s", weber_at_t0),
            (shape_data_T_tenth, "t = T/8", weber_at_T_tenth),
            (shape_data_T, "Terminal (T)", weber_at_T)
        ]
        
        for col_idx, (shape_data, time_label, weber_val) in enumerate(shapes_to_plot):
            ax_s = axs_shapes_flat[row_idx, col_idx]; 
            
            if shape_data is None or shape_data.get("d_horizontal_mm") is None or shape_data.get("d_vertical_mm") is None:
                ax_s.text(0.5, 0.5, "Data N/A", ha='center', va='center', fontsize=10, color='gray')
                ax_s.set_title(f"Initial {initial_d_mm:.1f}mm - {time_label}", fontsize=9)
                ax_s.set_aspect('equal', adjustable='box'); ax_s.set_xticks([]); ax_s.set_yticks([])
                continue

            d_horiz_mm = shape_data["d_horizontal_mm"]; d_vert_mm = shape_data["d_vertical_mm"]
            
            shape_patch = create_raindrop_patch(d_horiz_mm, d_vert_mm, weber_val) 
            shape_patch.set(facecolor='lightblue', edgecolor='darkblue', alpha=0.75) 
            ax_s.add_patch(shape_patch) 
            ax_s.set_title(f"Initial {initial_d_mm:.1f}mm - {time_label}\n(H:{d_horiz_mm:.2f} x V:{d_vert_mm:.2f} mm)", fontsize=9)
            
            ax_s.set_xlim(-fixed_axis_limit, fixed_axis_limit)
            ax_s.set_ylim(-fixed_axis_limit, fixed_axis_limit)
            ax_s.set_aspect('equal', adjustable='box') 
            
            if row_idx == num_target_drops -1 : ax_s.set_xlabel("Horizontal (mm)")
            else: ax_s.set_xticklabels([])
            if col_idx == 0: ax_s.set_ylabel("Vertical (mm)")
            else: ax_s.set_yticklabels([])
            ax_s.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) 


# -----------------------------------------------------------------------------
# 5. 빗방울 형태 변화 애니메이션 함수 그룹 (낙하 없이 형태만)
# -----------------------------------------------------------------------------
active_animations = [] 
def animate_raindrop_shape_only_evolution(result_data, save_gif=False, gif_filename="raindrop_shape_evolution.gif", fps=30):
    initial_d_mm = result_data["initial_diameter_mm"]; all_states = result_data["evolution_states"] 
    num_total_states = len(all_states)
    if num_total_states > TARGET_ANIMATION_FRAMES:
        indices = np.linspace(0, num_total_states - 1, TARGET_ANIMATION_FRAMES, dtype=int); sampled_states = [all_states[i] for i in indices]
    else: sampled_states = all_states
    fig_anim, ax_anim = plt.subplots(figsize=(7, 7)) 
    max_h_anim_mm = max(state[1] for state in sampled_states if state[1] is not None); max_v_anim_mm = max(state[2] for state in sampled_states if state[2] is not None)
    axis_limit_anim = max(max_h_anim_mm, max_v_anim_mm, initial_d_mm) * 1.2 / 2.0 
    if axis_limit_anim <= 1e-6 : axis_limit_anim = initial_d_mm * 1.2 / 2.0
    ax_anim.set_xlim(-axis_limit_anim, axis_limit_anim); ax_anim.set_ylim(-axis_limit_anim, axis_limit_anim); ax_anim.set_aspect('equal', adjustable='box'); ax_anim.set_xlabel("Horizontal Diameter (mm)"); ax_anim.set_ylabel("Vertical Diameter (mm)"); ax_anim.grid(True, linestyle=':', alpha=0.5)
    raindrop_shape_patch_anim = None; time_text_anim = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=10)
    fig_anim.suptitle(f"Raindrop Shape Evolution (Initial Diameter: {initial_d_mm:.1f} mm)", fontsize=14); plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    def update_anim_frame(frame_index): 
        nonlocal raindrop_shape_patch_anim; time_s, d_h_mm, d_v_mm, _, current_we_frame = sampled_states[frame_index] 
        if raindrop_shape_patch_anim: raindrop_shape_patch_anim.remove()
        raindrop_shape_patch_anim = create_raindrop_patch(d_h_mm, d_v_mm, current_we_frame) # create_raindrop_patch 사용
        raindrop_shape_patch_anim.set(facecolor='lightblue', edgecolor='darkblue', alpha=0.75) # Patch 속성 설정
        if raindrop_shape_patch_anim: ax_anim.add_patch(raindrop_shape_patch_anim) # Patch 추가
        time_text_anim.set_text(f"Time: {time_s:.3f} s\nWe: {current_we_frame:.2f}"); return raindrop_shape_patch_anim, time_text_anim
    def init_anim():
        nonlocal raindrop_shape_patch_anim; time_s_init, d_h_mm_init, d_v_mm_init, _, current_we_init = sampled_states[0]
        if raindrop_shape_patch_anim: raindrop_shape_patch_anim.remove()
        raindrop_shape_patch_anim = create_raindrop_patch(d_h_mm_init, d_v_mm_init, current_we_init) # create_raindrop_patch 사용
        raindrop_shape_patch_anim.set(facecolor='lightblue', edgecolor='darkblue', alpha=0.75) # Patch 속성 설정
        if raindrop_shape_patch_anim: ax_anim.add_patch(raindrop_shape_patch_anim) # Patch 추가
        time_text_anim.set_text(f"Time: {time_s_init:.3f} s\nWe: {current_we_init:.2f}"); return raindrop_shape_patch_anim, time_text_anim
    num_sampled_frames = len(sampled_states); interval_ms = 1000 / fps 
    current_ani = animation.FuncAnimation(fig_anim, update_anim_frame, frames=num_sampled_frames, init_func=init_anim, blit=False, repeat=False, interval=interval_ms); active_animations.append(current_ani) 
    if save_gif: current_ani.save(gif_filename, writer='pillow', fps=fps); plt.close(fig_anim) 
    else: plt.show() 
    return current_ani 


# -----------------------------------------------------------------------------
# 6. 메인 실행 함수
# -----------------------------------------------------------------------------
def main():
    raindrop_initial_diameters_mm = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] 
    simulation_results = []
    for diameter_mm in raindrop_initial_diameters_mm:
        result = simulate_single_raindrop(diameter_mm)
        simulation_results.append(result)
        
    plot_results(simulation_results)
    plot_raindrop_shapes(simulation_results, raindrop_initial_diameters_mm) 
    plot_overlay_terminal_shapes(simulation_results, INDENTATION_THRESHOLD_MM, MAX_DIAMETER_FOR_FULL_INDENT_MM)
    plt.show()
    
    animation_target_diameters_to_save_gif = raindrop_initial_diameters_mm 
    gif_output_folder = "raindrop_animations"
    if not os.path.exists(gif_output_folder):
        os.makedirs(gif_output_folder)

    results_dict = {r['initial_diameter_mm']: r for r in simulation_results}
    for anim_target_d_mm in animation_target_diameters_to_save_gif:
        anim_result_data = results_dict.get(anim_target_d_mm)
        
        if anim_result_data:
            gif_filename = os.path.join(gif_output_folder, f"shape_evolution_{anim_target_d_mm:.1f}mm.gif")
            animate_raindrop_shape_only_evolution(anim_result_data, save_gif=True, gif_filename=gif_filename, fps=30)

if __name__ == '__main__':
    main()
