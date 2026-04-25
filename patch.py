import re

with open('nexus_rl/server/nexus_rl_environment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. _get_agent_total_resources to include Vault
content = content.replace(
'''        return (
            agent.get("E_available", 0) + agent.get("E_locked", 0),
            agent.get("C_available", 0) + agent.get("C_locked", 0)
        )''',
'''        return (
            agent.get("E_available", 0) + agent.get("E_locked", 0) + agent.get("E_vault", 0),
            agent.get("C_available", 0) + agent.get("C_locked", 0) + agent.get("C_vault", 0)
        )'''
)

# 2. Add WORK and VAULT and SIGNAL handling in step()
anchor = '''        if not validation_errors and action.action_type == "PROPOSE":
            proposal_key = f"{agent_0_id}->{action.target_id}"
            # INVENTORY LOCKING: Lock the proposed energy to prevent double-spending
            if self._lock_resources(agent_0_id, action.offer_E, 0):
                self.active_proposals[proposal_key] = {
                    "action": action,
                    "created_step": self._state.step_count
                }
            else:
                # Insufficient available energy, add to validation errors
                validation_errors.append(f"Insufficient available energy to lock {action.offer_E}")'''

replacement = '''        if not validation_errors:
            # Decay immunity per step
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].get("tax_immunity", 0) > 0:
                    self.agents[agent_id]["tax_immunity"] -= 1

            if action.action_type == "PROPOSE":
                proposal_key = f"{agent_0_id}->{action.target_id}"
                
                # Check commitment
                agent = self.agents[agent_0_id]
                if agent.get("commitment_target") == action.target_id:
                    if action.offer_E < agent.get("commitment_E", 0) or action.request_C > agent.get("commitment_C", 0):
                        agent["collateral"] = max(0.0, agent["collateral"] - 2.0)  # Broken promise penalty
                
                # Listing fee & stake
                if agent["collateral"] >= 1.0:
                    agent["collateral"] -= 1.0
                    
                    if self._lock_resources(agent_0_id, action.offer_E, 0):
                        self.active_proposals[proposal_key] = {
                            "action": action,
                            "created_step": self._state.step_count
                        }
                    else:
                        validation_errors.append(f"Insufficient available energy to lock {action.offer_E}")
                        agent["collateral"] += 1.0 # refund
                else:
                    validation_errors.append("Insufficient collateral to propose")
            
            elif action.action_type == "WORK":
                # Spend 5E to get 2C, or 5C to get 2E, gives 5 steps tax immunity
                agent = self.agents[agent_0_id]
                if action.offer_E > 0 and agent["E_available"] >= 5:
                    agent["E_available"] -= 5
                    agent["C_available"] += 2
                    agent["tax_immunity"] = 5
                elif action.request_C > 0 and agent["C_available"] >= 5:
                    agent["C_available"] -= 5
                    agent["E_available"] += 2
                    agent["tax_immunity"] = 5
                    
            elif action.action_type == "VAULT":
                agent = self.agents[agent_0_id]
                if action.offer_E > 0 and agent["E_available"] >= action.offer_E:
                    agent["E_available"] -= action.offer_E
                    agent["E_vault"] += action.offer_E
                if action.request_C > 0 and agent["C_available"] >= action.request_C:
                    agent["C_available"] -= action.request_C
                    agent["C_vault"] += action.request_C
                    
            elif action.action_type == "SIGNAL":
                agent = self.agents[agent_0_id]
                agent["commitment_E"] = action.signal_offer_E
                agent["commitment_C"] = action.signal_request_C
                agent["commitment_target"] = action.target_id'''

content = content.replace(anchor, replacement)

# 3. Add Time Tax in step() expiration
anchor2 = '''        for key in expired_proposals:
            # INVENTORY LOCKING: Unlock resources when proposal expires
            proposal_data = self.active_proposals[key]
            action = proposal_data.get("action")
            if action and action.action_type == "PROPOSE":
                proposer_id = int(key.split("->")[0])
                self._unlock_resources(proposer_id, action.offer_E)
            del self.active_proposals[key]'''

replacement2 = '''        for key in expired_proposals:
            proposal_data = self.active_proposals[key]
            action = proposal_data.get("action")
            if action and action.action_type == "PROPOSE":
                proposer_id = int(key.split("->")[0])
                self._unlock_resources(proposer_id, action.offer_E)
                # Time Tax for expiring proposal
                self.agents[proposer_id]["collateral"] = max(0.0, self.agents[proposer_id]["collateral"] - 0.5)
            del self.active_proposals[key]'''

content = content.replace(anchor2, replacement2)

# 4. Give collateral back on fulfilled
anchor3 = '''            self.agents[proposer_id]["E_locked"] -= offer_E
            self.agents[proposer_id]["C_available"] += request_C
            
            self.agents[target_id]["E_available"] += offer_E
            self.agents[target_id]["C_available"] -= request_C'''

replacement3 = '''            self.agents[proposer_id]["E_locked"] -= offer_E
            self.agents[proposer_id]["C_available"] += request_C
            
            self.agents[target_id]["E_available"] += offer_E
            self.agents[target_id]["C_available"] -= request_C
            
            # Return partial collateral stake to proposer since trade succeeded
            self.agents[proposer_id]["collateral"] += 0.9'''

content = content.replace(anchor3, replacement3)

# 5. Modify Reward to include Net Collateral change
anchor4 = '''        # Social Welfare Reward Function
        # R = Agent 0's gain + λ * (System welfare gain)
        agent_0_reward_component = self.config.REWARD_WEIGHT_UTILITY * delta_utility_agent_0
        system_welfare_component = self.config.ALTRUISM_COEFFICIENT * total_delta_utility_others
        reward = agent_0_reward_component + system_welfare_component'''

replacement4 = '''        # Social Welfare Reward Function
        agent_0_reward_component = self.config.REWARD_WEIGHT_UTILITY * delta_utility_agent_0
        system_welfare_component = self.config.ALTRUISM_COEFFICIENT * total_delta_utility_others
        
        # Penalize loss of collateral
        delta_collateral = self.agents[agent_0_id]["collateral"] - self.previous_collateral if hasattr(self, 'previous_collateral') else 0.0
        self.previous_collateral = self.agents[agent_0_id]["collateral"]
        
        collateral_component = min(0.0, delta_collateral) # Only penalize loss, don't reward hoarding
        
        reward = agent_0_reward_component + system_welfare_component + collateral_component'''

content = content.replace(anchor4, replacement4)

# 6. Apply Tax Immunity in Shocks
anchor5 = '''                # ASYMMETRIC: Rich agents lose more (1.5x multiplier if above average)
                wealth_ratio = agent_energy / avg_energy if avg_energy > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                energy_loss = int(agent_energy * min(asymmetric_loss_pct, 0.5))  # Cap at 50%'''

replacement5 = '''                # Check Tax Immunity (from WORK)
                if agent.get("tax_immunity", 0) > 0:
                    continue  # Immune to shock
                    
                # ASYMMETRIC: Rich agents lose more (1.5x multiplier if above average)
                wealth_ratio = agent_energy / avg_energy if avg_energy > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                energy_loss = int(agent_energy * min(asymmetric_loss_pct, 0.5))  # Cap at 50%'''
content = content.replace(anchor5, replacement5)

anchor6 = '''                # ASYMMETRIC: Rich agents lose more
                wealth_ratio = agent_compute / avg_compute if avg_compute > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                compute_loss = int(agent_compute * min(asymmetric_loss_pct, 0.5))'''

replacement6 = '''                # Check Tax Immunity (from WORK)
                if agent.get("tax_immunity", 0) > 0:
                    continue  # Immune to shock
                    
                # ASYMMETRIC: Rich agents lose more
                wealth_ratio = agent_compute / avg_compute if avg_compute > 0 else 1.0
                asymmetric_loss_pct = loss_pct * (1.0 + max(0, wealth_ratio - 1.0) * 0.5)
                
                compute_loss = int(agent_compute * min(asymmetric_loss_pct, 0.5))'''
content = content.replace(anchor6, replacement6)

with open('nexus_rl/server/nexus_rl_environment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
