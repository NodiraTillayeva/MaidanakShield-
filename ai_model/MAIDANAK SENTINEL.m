close all; clear; clc;
% 
% potassium_nitrate = 65;
% sucrose = 35;
% 
% P_atm = 10;
% P_MPa = P_atm * 0.000101325;
% r = 2.84 * (P_MPa^0.4);
% 
% rho_g_cm = 1.89;
% rho_g_mm = rho_g_cm / 1000;  % 1 cm^3 = 1000 mm^3
% nozzle_throat_diameter = 6.75; %mm
% chamber_diameter = 76.2;  %mm
% 
% mass_flow_rate = (pi * (nozzle_throat_diameter / 2)^2) * r * rho_g_mm;
% Pte_Ptc = 0.95; 
% h = 0; % sealevel
% thrust_at_sealevelstaticconditions = mass_flow_rate * Pte_Ptc * (P_MPa)
% units = thrust_at_sealevelstaticconditions * 1000 % Convert to Newtons;
% g0 = 9.81; % Acceleration due to gravity in m/s^2
% massflowratecm = mass_flow_rate * 100; % Convert to cm^3/s
% massflowratem = massflowratecm / 1000000 % Convert to m^3/s
% Isp = thrust_at_sealevelstaticconditions / (massflowratem * g0) % units?
% fprintf('Thrust at sea level: %.2f N\n', units);
% fprintf('Specific Impulse (Isp): %.2f s\n', Isp); 
% assess_solution = Isp * g0; % Calculate the effective exhaust velocity
% effective_exhaust_velocity = assess_solution; 
% fprintf('Effective Exhaust Velocity: %.2f m/s\n', effective_exhaust_velocity);
% commentonsolution = 'The calculated effective exhaust velocity indicates the efficiency of the propulsion system.';
% % Calculate the total impulse over a given burn time
% burn_time = 60; % seconds
% total_impulse = thrust_at_sealevelstaticconditions * burn_time; % in Newton-seconds
% fprintf('Total Impulse: %.2f N·s\n', total_impulse);

% given
dry_mass_spacecraft = 100000;% kg
P_thrustchamper = 2; % Mpa
T_thrustchamper= 2500; %K
o_f = 4;% oxygen to fuel ratio
A_nozzle_throat = 0.020;
A_expansionnozzle = 100;
cd= 1; %discharge coefficient
y = 1.233; % ratio of specific heats
molecular_weight = 26.35; % g/mol 
R = 8314 / molecular_weight; % Specific gas constant in J/(kg·K)
exhaust_velocity = sqrt(2 * y * R * T_thrustchamper / (y - 1)); % Calculate exhaust gas velocity
fprintf('Exhaust Gas Velocity: %.2f m/s\n', exhaust_velocity);
delta_V =500; % m/s 
amount_of_propellant = (dry_mass_spacecraft * delta_V) / exhaust_velocity; % Calculate the amount of propellant needed
fprintf('Amount of Propellant Needed: %.2f kg\n', amount_of_propellant); %assume drymass is the payload; 
payload_fraction = amount_of_propellant / (dry_mass_spacecraft + amount_of_propellant);
fprintf('Payload Fraction at the Beginning of Burn: %.2f\n', payload_fraction); 
% Calculate the payload fraction at the end of the burn
final_mass_spacecraft = dry_mass_spacecraft + amount_of_propellant - amount_of_propellant; % Assuming all propellant is used
final_payload_fraction = amount_of_propellant / final_mass_spacecraft;
fprintf('Payload Fraction at the End of Burn: %.2f\n', final_payload_fraction);% 
which produces more thrust out of these 3: hybrid rocket motor, ion thruster, solid rocket motor  - ?
initial_total_mass = dry_mass_spacecraft + amount_of_propellant;
thrust_to_weight_ratio_initial = thrust_at_sealevelstaticconditions / (initial_total_mass * g0);
fprintf('Thrust-to-Weight Ratio at the Beginning of Burn: %.2f\n', thrust_to_weight_ratio_initial);

